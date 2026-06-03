"""Price structured products as weighted sums of already-priced components."""

from __future__ import annotations

from typing import cast

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.core.columns import cols
from schenberg.domain.schemas.position import InstrumentPrice
from schenberg.domain.schemas.structure import StructureLeg

L = cols(StructureLeg)
PX = cols(InstrumentPrice)


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
        left_on=[L.component_instrument_type.name, L.component_instrument_id.name],
        right_on=[PX.instrument_type.name, PX.instrument_id.name],
        how="left",
    ).with_columns(component_value=L.side.expr() * L.quantity.expr() * PX.price.expr())

    result = (
        priced_components.group_by(L.structure_id.name)
        .agg(price=pl.col("component_value").sum())
        .with_columns(instrument_type=pl.lit(structure_type))
        .rename({L.structure_id.name: PX.instrument_id.name})
        .select(PX.instrument_type.name, PX.instrument_id.name, PX.price.name)
    )

    return cast(LazyFrame[InstrumentPrice], result)
