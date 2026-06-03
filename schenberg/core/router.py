"""Lazy row routing for heterogeneous instrument pricers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

import polars as pl

from schenberg.core.columns import ColumnRef, RoutePredicate
from schenberg.market_data.snapshot import MarketSnapshot


class Pricer(Protocol):
    def compute_for(
        self,
        lf: pl.LazyFrame,
        *,
        market: MarketSnapshot | None = None,
        output_profile: str = "pricing",
    ) -> pl.LazyFrame: ...


P = TypeVar("P", bound=Pricer)


@dataclass(slots=True)
class Router:
    route_columns: tuple[ColumnRef, ...]
    cases: list[tuple[tuple[RoutePredicate, ...], Pricer]] = field(default_factory=list)
    fallback: Pricer | None = None

    @classmethod
    def by(cls, *columns: ColumnRef) -> Router:
        if not columns:
            raise ValueError("Router.by(...) requires at least one route column")
        return cls(route_columns=tuple(columns))

    def default(self, pricer: Pricer) -> Router:
        self.fallback = pricer
        return self

    def register(
        self,
        *predicates: RoutePredicate,
    ) -> Callable[[Callable[[], P]], P]:
        def decorator(builder: Callable[[], P]) -> P:
            pricer = builder()
            self.cases.append((tuple(predicates), pricer))
            return pricer

        return decorator

    def compute_for(
        self,
        lf: pl.LazyFrame,
        *,
        market: MarketSnapshot | None = None,
        output_profile: str = "pricing",
    ) -> pl.LazyFrame:
        parts: list[pl.LazyFrame] = []
        matched = pl.lit(False)

        for predicates, pricer in self.cases:
            condition = self._and(predicates)
            matched = matched | condition

            parts.append(
                pricer.compute_for(
                    lf.filter(condition),
                    market=market,
                    output_profile=output_profile,
                )
            )

        if self.fallback is not None:
            parts.append(
                self.fallback.compute_for(
                    lf.filter(~matched),
                    market=market,
                    output_profile=output_profile,
                )
            )

        if not parts:
            raise ValueError("router has no registered cases and no fallback")

        return pl.concat(parts, how="diagonal_relaxed")

    @staticmethod
    def _and(predicates: tuple[RoutePredicate, ...]) -> pl.Expr:
        if not predicates:
            return pl.lit(True)

        condition = pl.lit(True)
        for predicate in predicates:
            condition = condition & predicate.expr()
        return condition
