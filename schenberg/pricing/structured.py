"""Price structured products as weighted sums of already-priced components.

A structured product is a table of component legs (:class:`StructureLeg`). Each
leg's *exposure* is ``side * quantity`` — position direction that lives here, at
the structure level, never in a pricing graph. Component prices are looked up by a
join; the per-structure roll-up is a :class:`~schenberg.core.fold.Fold` so the
"group by structure and sum contributions" semantics live in one place.
"""

from __future__ import annotations

from typing import cast

import pandera.polars as pa
from pandera.typing.polars import LazyFrame

from schenberg.core.columns import cols
from schenberg.core.fold import Fold, lit_, sum_
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
    summed per ``structure_id`` with a :class:`Fold`. The result has the same shape
    as any other ``InstrumentPrice`` frame and can be passed to ``with_prices``
    directly.
    """
    contributions = (
        structure_legs.join(
            component_prices,
            left_on=[L.component_instrument_type.name, L.component_instrument_id.name],
            right_on=[PX.instrument_type.name, PX.instrument_id.name],
            how="left",
        )
        .with_columns(component_value=L.side.expr() * L.quantity.expr() * PX.price.expr())
        .rename({L.structure_id.name: PX.instrument_id.name})
    )

    fold = (
        Fold("structure_price", input_schema=StructureLeg)
        .by(PX.instrument_id)
        .returns(
            InstrumentPrice,
            instrument_type=lit_(structure_type),
            price=sum_("component_value"),
        )
    )
    return cast(LazyFrame[InstrumentPrice], fold.compute(contributions))
