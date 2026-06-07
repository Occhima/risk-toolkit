"""Declarative attachable market dependencies."""

from __future__ import annotations

from collections.abc import Callable
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
    :meth:`with_output` so ``g.market(rate=...)`` can name it from a keyword.
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
class MarketRead:
    """A market read that knows *what* to read but not yet *where* to write it.

    The output is supplied later — ``g.market`` names it from the keyword — by
    calling :meth:`as_output`, which builds the concrete
    :class:`MarketDependency`. This is the Reader half of the market layer: a
    delayed dependency awaiting its environment binding.
    """

    build: Callable[[str], MarketDependency]

    def as_output(self, output: str) -> MarketDependency:
        return self.build(output)


# Either a fully built dependency (output fixed) or a delayed read awaiting an
# output name. ``g.market`` accepts both and finalizes a read.
AttachableMarket = MarketDependency | MarketRead


def finalize_market(read: AttachableMarket, output: str) -> MarketDependency:
    """Bind ``output`` onto a read or rename a built dependency to it."""
    if isinstance(read, MarketRead):
        return read.as_output(output)
    return read.with_output(output)


@dataclass(frozen=True, slots=True)
class MarketRequirement:
    """A keyed left-join read: pull ``outputs`` columns from ``table`` on ``on``.

    A spec hands back a requirement whose output ``g.market(rate=...)`` names from
    its keyword via :meth:`with_output`. A multi-output requirement (a join that
    writes several columns at once) cannot be renamed by keyword; ``g.market``
    keeps its own output names instead.
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
                f"{sorted(self.outputs.values())}); pass it to g.market(), which keeps "
                f"its own output names"
            )
        (value_col,) = self.outputs
        return replace(self, outputs={value_col: output})

    def attach(self, lf: pl.LazyFrame, snapshot: MarketSnapshot) -> pl.LazyFrame:
        """Attach market columns by a left join."""
        src = snapshot.source(self.table).data
        # Keep original column names in right — rename after joining to avoid
        # duplicate-column errors when an output name matches a join key.
        right = src.select([*self.right_keys, *self.outputs.keys()])

        existing = set(lf.collect_schema().names())
        output_names = set(self.outputs.values())
        left_key_set = set(self.left_keys)

        non_key_collisions = sorted((output_names - left_key_set) & existing)
        if non_key_collisions:
            raise ValueError(
                f"market requirement for table {self.table!r} would overwrite existing "
                f"column(s): {non_key_collisions}. Rename the market role/output or drop "
                "the columns explicitly before binding."
            )

        result = lf.join(
            right,
            left_on=list(self.left_keys),
            right_on=list(self.right_keys),
            how="left",
        )

        # Drop left join-key columns that are overwritten by an output.
        key_overwrites = sorted(left_key_set & output_names & existing)
        if key_overwrites:
            result = result.drop(key_overwrites)

        renames = {
            val_col: out_col for val_col, out_col in self.outputs.items() if val_col != out_col
        }
        if renames:
            result = result.rename(renames)

        return result
