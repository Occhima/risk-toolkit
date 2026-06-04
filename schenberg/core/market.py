"""Declarative attachable market dependencies."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

import polars as pl

from schenberg.core.columns import ColumnSet

if TYPE_CHECKING:
    from schenberg.market_data.snapshot import MarketSnapshot


class MarketDependency(Protocol):
    """Market data a graph can attach before its formulas run.

    Most dependencies are a simple keyed join (:class:`MarketRequirement`); a few,
    such as volatility surfaces, interpolate. Both expose the same graph-facing
    shape, and both can have their (single) output column renamed by
    :meth:`with_output` so ``FormulaGraph.for_market`` can name it from a keyword.
    """

    table: str
    outputs: dict[str, str]

    @property
    def left_keys(self) -> tuple[str, ...]: ...

    @property
    def right_keys(self) -> tuple[str, ...]: ...

    def with_output(self, output: str) -> MarketDependency: ...

    def attach(self, lf: pl.LazyFrame, snapshot: MarketSnapshot) -> pl.LazyFrame: ...


@dataclass(frozen=True, slots=True)
class MarketRequirement:
    """A keyed left-join read: pull ``outputs`` columns from ``table`` on ``on``.

    A spec hands back a requirement whose single output defaults to the value
    column's own name; ``FormulaGraph.for_market(rate=...)`` then renames it to the
    keyword via :meth:`with_output`. Multi-output requirements (a join that writes
    several columns at once) are attached through ``uses_market`` and cannot be
    renamed by keyword.
    """

    table: str
    on: ColumnSet
    outputs: dict[str, str]

    @property
    def left_keys(self) -> tuple[str, ...]:
        return self.on.left_keys

    @property
    def right_keys(self) -> tuple[str, ...]:
        return self.on.right_keys

    def with_output(self, output: str) -> MarketRequirement:
        if len(self.outputs) != 1:
            raise ValueError(
                f"cannot rename a multi-output requirement (table {self.table!r} writes "
                f"{sorted(self.outputs.values())}); attach it with uses_market(...)"
            )
        (value_col,) = self.outputs
        return replace(self, outputs={value_col: output})

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
