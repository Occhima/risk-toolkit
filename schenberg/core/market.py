"""Declarative attachable market dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

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


# Requirement helpers for the swap legs. New market-data specs live under
# schenberg.market_data.* and should be preferred for new instruments.
def curve(
    *identity: str,
    indexer_col: str = "id_indexador",
    tenor_col: str = "payment_days",
    outputs: dict[str, str] | None = None,
) -> MarketRequirement:
    out = {name: name for name in identity}
    if outputs:
        out.update(outputs)
    return MarketRequirement(
        table="curves",
        on=ColumnSet.from_pairs(
            (indexer_col, "id_indexador"),
            (tenor_col, "tenor_days"),
        ),
        outputs=out,
    )


def fixing(
    *, indexer_col: str = "id_indexador", date_col: str = "base_date", output: str = "base_index"
) -> MarketRequirement:
    return MarketRequirement(
        table="fixings",
        on=ColumnSet.from_pairs(
            (indexer_col, "id_indexador"),
            (date_col, "fixing_date"),
        ),
        outputs={"fixing_value": output},
    )


def projected_index(
    *,
    indexer_col: str = "id_indexador",
    tenor_col: str = "payment_days",
    output: str = "projected_index",
) -> MarketRequirement:
    return MarketRequirement(
        table="projected_indexes",
        on=ColumnSet.from_pairs(
            (indexer_col, "id_indexador"),
            (tenor_col, "tenor_days"),
        ),
        outputs={"projected_index": output},
    )
