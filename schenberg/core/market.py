"""Declarative attachable market dependencies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import polars as pl

from schenberg.core.columns import ColumnSet

if TYPE_CHECKING:
    from schenberg.market_data.snapshot import MarketSnapshot


class MarketDependency(Protocol):
    """Small protocol for market data a graph can attach before formulas.

    Most dependencies are simple joins. A few, such as volatility surfaces,
    interpolate. Both expose the same graph-facing shape.
    """

    table: str
    outputs: dict[str, str]

    @property
    def left_keys(self) -> tuple[str, ...]: ...

    @property
    def right_keys(self) -> tuple[str, ...]: ...

    def attach(self, lf: pl.LazyFrame, snapshot: MarketSnapshot) -> pl.LazyFrame: ...


@dataclass(frozen=True, slots=True)
class MarketRead:
    """A market read whose output column is not yet decided.

    Market specs return a ``MarketRead`` when no ``output`` is passed: the read
    knows its source and join keys but waits for ``FormulaGraph.for_market`` to
    name the output column from the keyword it is bound to::

        graph.for_market(rate=CURVES.value("zero_rate", ...))  # -> output "rate"
    """

    build: Callable[[str], Any]

    def as_output(self, output: str) -> MarketDependency:
        return self.build(output)


@dataclass(frozen=True, slots=True)
class MarketRequirement:
    table: str
    on: ColumnSet
    outputs: dict[str, str]

    @property
    def left_keys(self) -> tuple[str, ...]:
        return self.on.left_keys

    @property
    def right_keys(self) -> tuple[str, ...]:
        return self.on.right_keys

    def attach(self, lf: pl.LazyFrame, snapshot: MarketSnapshot) -> pl.LazyFrame:
        """Attach market columns by a left join."""
        src = snapshot.source(self.table).data
        right = src.select([*self.right_keys, *self.outputs.keys()]).rename(self.outputs)
        output_columns = set(self.outputs.values())
        droppable = output_columns - set(self.left_keys)
        collisions = sorted(droppable & set(lf.collect_schema().names()))
        if collisions:
            lf = lf.drop(collisions)

        return lf.join(
            right,
            left_on=list(self.left_keys),
            right_on=list(self.right_keys),
            how="left",
        )
