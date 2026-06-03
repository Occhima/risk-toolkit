"""Lazy row routing for heterogeneous instrument pricers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeVar

import polars as pl

from schenberg.core.columns import ColumnRef, RoutePredicate

if TYPE_CHECKING:
    from schenberg.market_data.snapshot import MarketSnapshot


class Pricer(Protocol):
    def compute(
        self,
        frame: pl.LazyFrame,
        *,
        market: MarketSnapshot | None = None,
        view: str = "result",
    ) -> pl.LazyFrame: ...


P = TypeVar("P", bound=Pricer)


@dataclass(slots=True)
class Router:
    """Dispatch rows to per-case pricers by predicate.

    Build with :meth:`on` over one or more route columns, then register cases
    with the :meth:`case` decorator (equality on the route columns) or the
    :meth:`when` decorator (arbitrary predicates). Unmatched rows fall to
    :meth:`default`.
    """

    route_columns: tuple[ColumnRef, ...]
    cases: list[tuple[tuple[RoutePredicate | pl.Expr, ...], Pricer]] = field(default_factory=list)
    fallback: Pricer | None = None

    @classmethod
    def on(cls, *columns: ColumnRef) -> Router:
        if not columns:
            raise ValueError("Router.on(...) requires at least one route column")
        return cls(route_columns=tuple(columns))

    def default(self, pricer: Pricer) -> Router:
        self.fallback = pricer
        return self

    def case(self, *values: object) -> Callable[[Callable[[], P]], P]:
        """Register a case by value, one per route column.

        ``router.case(OptionModel.GENERALIZED, OptionKind.CALL)`` builds the
        equality predicates ``col == value`` against the route columns.
        """
        if len(values) != len(self.route_columns):
            raise ValueError(
                f"case expects {len(self.route_columns)} value(s) for route columns "
                f"{[c.name for c in self.route_columns]}, got {len(values)}"
            )
        predicates = tuple(
            column == value for column, value in zip(self.route_columns, values, strict=True)
        )
        return self.when(*predicates)

    def when(
        self,
        *predicates: RoutePredicate | pl.Expr,
    ) -> Callable[[Callable[[], P]], P]:
        """Register a case by explicit predicates (supports complex conditions)."""

        def decorator(builder: Callable[[], P]) -> P:
            pricer = builder()
            self.cases.append((tuple(predicates), pricer))
            return pricer

        return decorator

    def compute(
        self,
        frame: pl.LazyFrame,
        *,
        market: MarketSnapshot | None = None,
        view: str = "result",
    ) -> pl.LazyFrame:
        parts: list[pl.LazyFrame] = []
        matched = pl.lit(False)

        for predicates, pricer in self.cases:
            condition = self._and(predicates)
            matched = matched | condition

            parts.append(
                pricer.compute(
                    frame.filter(condition),
                    market=market,
                    view=view,
                )
            )

        if self.fallback is not None:
            parts.append(
                self.fallback.compute(
                    frame.filter(~matched),
                    market=market,
                    view=view,
                )
            )

        if not parts:
            raise ValueError("router has no registered cases and no fallback")

        return pl.concat(parts, how="diagonal_relaxed")

    @staticmethod
    def _and(predicates: tuple[RoutePredicate | pl.Expr, ...]) -> pl.Expr:
        if not predicates:
            return pl.lit(True)

        condition = pl.lit(True)
        for predicate in predicates:
            expr = predicate.expr() if isinstance(predicate, RoutePredicate) else predicate
            condition = condition & expr
        return condition
