"""Shock: an endomorphism on :class:`MarketSnapshot`.

A shock is a pure function ``MarketSnapshot -> MarketSnapshot``. It never mutates
its argument: it returns a new snapshot with one or more sources transformed,
sharing the schemas and untouched sources of the original. Because shocks are
endomorphisms they **compose** — :meth:`then` and :meth:`compose` chain them, and
:meth:`identity` is the unit — which is exactly what a scenario set (parallel
bumps, curve twists, vol shifts) needs.

Shocks are usually built from a :class:`~schenberg.market_data.path.MarketPath`
(``MarketPath("curves").column("zero_rate").modify(lambda r: r + 1e-4)``) or from
the convenience constructors here. Every shock can :meth:`explain` itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource

ShockFn = Callable[[MarketSnapshot], MarketSnapshot]


@dataclass(frozen=True, slots=True)
class Shock:
    """A named, composable transform of a market snapshot."""

    name: str
    apply_fn: ShockFn

    def __call__(self, market: MarketSnapshot) -> MarketSnapshot:
        return self.apply_fn(market)

    def apply(self, market: MarketSnapshot) -> MarketSnapshot:
        """Alias for calling the shock — reads well at the call site."""
        return self.apply_fn(market)

    def then(self, other: Shock) -> Shock:
        """Sequential composition: apply ``self`` then ``other``."""
        return Shock(f"{self.name} >> {other.name}", lambda m: other(self(m)))

    @staticmethod
    def compose(*shocks: Shock) -> Shock:
        """Compose shocks left-to-right into one shock (identity if empty)."""
        if not shocks:
            return Shock.identity()
        name = " >> ".join(s.name for s in shocks)

        def run(market: MarketSnapshot) -> MarketSnapshot:
            for shock in shocks:
                market = shock(market)
            return market

        return Shock(name, run)

    @staticmethod
    def identity() -> Shock:
        """The no-op shock — the unit of :meth:`compose` / :meth:`then`."""
        return Shock("identity", lambda m: m)

    def info(self) -> dict[str, object]:
        return {"name": self.name}

    def explain(self) -> str:
        return f"Shock {self.name}\n  - MarketSnapshot -> MarketSnapshot (endomorphism)"


def modify_source(
    market: MarketSnapshot,
    source_name: str,
    column: str,
    fn_or_expr: Callable[[pl.Expr], pl.Expr] | pl.Expr,
) -> MarketSnapshot:
    """Return a new snapshot with ``column`` of ``source_name`` transformed.

    ``fn_or_expr`` is either a callable ``pl.Expr -> pl.Expr`` applied to the
    column, or a ready ``pl.Expr``. The source schema and all other sources are
    preserved; the original snapshot is untouched.
    """
    source = market.source(source_name)
    new_expr = fn_or_expr if isinstance(fn_or_expr, pl.Expr) else fn_or_expr(pl.col(column))
    bumped = MarketSource(
        name=source.name,
        data=source.data.with_columns(new_expr.alias(column)),
        schema=source.schema,
    )
    return market.with_source(bumped)


def curve_parallel_shift(
    *, source: str = "curves", column: str = "zero_rate", shift: float = 1e-4
) -> Shock:
    """A parallel additive shift of a curve column (e.g. +1bp on every zero rate)."""
    return Shock(
        f"{source}.{column} += {shift}",
        lambda m: modify_source(m, source, column, lambda r: r + shift),
    )


def vol_parallel_shift(
    *, source: str = "vol_surface", column: str = "implied_vol", shift: float = 0.01
) -> Shock:
    """A parallel additive shift of a vol-surface column (e.g. +1 vol point)."""
    return Shock(
        f"{source}.{column} += {shift}",
        lambda m: modify_source(m, source, column, lambda v: v + shift),
    )
