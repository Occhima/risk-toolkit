"""MarketPath: a lens-lite onto a market source/column.

A :class:`MarketPath` focuses on one market source (and optionally one of its
columns) inside a :class:`MarketSnapshot`. It is an optic in spirit — a *getter*
(:meth:`get`) and a *modifier* (:meth:`modify` / :meth:`set_`) — kept deliberately
small: no profunctor machinery, just enough to read a source, build a
:class:`~schenberg.market_data.shocks.Shock`, or eagerly transform a snapshot.

::

    shock = MarketPath("curves").column("zero_rate").modify(lambda r: r + 1e-4)
    stressed = market.apply(shock)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import polars as pl

from schenberg.market_data.shocks import Shock, modify_source

if TYPE_CHECKING:
    from schenberg.market_data.snapshot import MarketSnapshot

Modifier = Callable[[pl.Expr], pl.Expr] | pl.Expr


@dataclass(frozen=True, slots=True)
class MarketPath:
    """Focus on a market source, and optionally one column of it."""

    source: str
    column_name: str | None = None

    def column(self, name: str) -> MarketPath:
        """Narrow the focus to a single column of the source."""
        return replace(self, column_name=name)

    def get(self, market: MarketSnapshot) -> pl.LazyFrame:
        """Read the focused data: the whole source frame, or one-column projection
        when a column is focused. Stays lazy."""
        data = market.source(self.source).data
        if self.column_name is None:
            return data
        return data.select(self.column_name)

    def set_(self, market: MarketSnapshot, fn_or_expr: Modifier) -> MarketSnapshot:
        """Eagerly transform the focused column, returning a new snapshot."""
        if self.column_name is None:
            raise ValueError(f"path {self.source!r} has no focused column; call .column(...)")
        return modify_source(market, self.source, self.column_name, fn_or_expr)

    def modify(self, fn_or_expr: Modifier) -> Shock:
        """Build a :class:`Shock` that transforms the focused column.

        ``fn_or_expr`` is a callable ``pl.Expr -> pl.Expr`` (applied to the column)
        or a ready ``pl.Expr``. The shock is composable and non-mutating.
        """
        if self.column_name is None:
            raise ValueError(f"path {self.source!r} has no focused column; call .column(...)")
        column = self.column_name
        return Shock(
            f"{self.source}.{column} modified",
            lambda m: modify_source(m, self.source, column, fn_or_expr),
        )

    def as_shock(self, fn_or_expr: Modifier) -> Shock:
        """Alias for :meth:`modify`."""
        return self.modify(fn_or_expr)
