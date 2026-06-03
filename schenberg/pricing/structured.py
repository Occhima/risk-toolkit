"""Price structured products as weighted sums of already-priced components."""

from __future__ import annotations

from typing import cast

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.domain.schemas.position import InstrumentPrice
from schenberg.domain.schemas.structure import StructureLeg


@pa.check_types(lazy=True)
def price_structures(
    structure_legs: LazyFrame[StructureLeg],
    component_prices: LazyFrame[InstrumentPrice],
    *,
    structure_type: str = "STRUCTURE",
) -> LazyFrame[InstrumentPrice]:
    """Aggregate component prices into structure-level prices.

    Each leg's contribution is ``side * quantity * price``; contributions are
    summed per ``structure_id``.  The result has the same shape as any other
    ``InstrumentPrice`` frame and can be passed to ``with_prices`` directly.
    """
    priced_components = structure_legs.join(
        component_prices,
        left_on=["component_instrument_type", "component_instrument_id"],
        right_on=["instrument_type", "instrument_id"],
        how="left",
    ).with_columns(component_value=pl.col("side") * pl.col("quantity") * pl.col("price"))

    result = (
        priced_components.group_by("structure_id")
        .agg(price=pl.col("component_value").sum())
        .with_columns(instrument_type=pl.lit(structure_type))
        .rename({"structure_id": "instrument_id"})
        .select("instrument_type", "instrument_id", "price")
    )

    return cast(LazyFrame[InstrumentPrice], result)
