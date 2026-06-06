"""Price structured products as weighted sums of already-valued components.

A structured product is a table of component legs (:class:`StructureLeg`). Each
leg's *exposure* is ``side * quantity`` — position direction that lives here, at
the structure level, never in a pricing graph. Component values are looked up by a
join; the per-structure roll-up is a :class:`~schenberg.core.fold.Fold` so the
"group by structure and sum contributions" semantics live in one place. The
result is itself an :class:`InstrumentValue`, so it feeds ``position_value``
directly alongside the atomic instruments.
"""

from __future__ import annotations

from typing import cast

import pandera.polars as pa
from pandera.typing.polars import LazyFrame

from schenberg.core.columns import cols
from schenberg.core.fold import Fold, first_, lit_, sum_
from schenberg.domain.schemas.position import InstrumentValue
from schenberg.domain.schemas.structure import StructureLeg

L = cols(StructureLeg)
IV = cols(InstrumentValue)


@pa.check_types(lazy=True)
def price_structures(
    structure_legs: LazyFrame[StructureLeg],
    component_values: LazyFrame[InstrumentValue],
    *,
    structure_type: str = "STRUCTURE",
) -> LazyFrame[InstrumentValue]:
    """Aggregate component instrument values into structure-level values.

    Each leg's contribution is ``side * quantity * value``; contributions are
    summed per ``structure_id`` with a :class:`Fold`. The result is an
    :class:`InstrumentValue` (the structure's currency is carried from its
    components), so it concatenates with atomic values and feeds ``position_value``
    directly.
    """
    contributions = (
        structure_legs.join(
            component_values,
            left_on=[L.component_instrument_type.name, L.component_instrument_id.name],
            right_on=[IV.instrument_type.name, IV.instrument_id.name],
            how="left",
        )
        .with_columns(component_value=L.side.expr() * L.quantity.expr() * IV.value.expr())
        .rename({L.structure_id.name: IV.instrument_id.name})
    )

    fold = (
        Fold("structure_value", input_schema=StructureLeg)
        .by(IV.instrument_id)
        .returns(
            InstrumentValue,
            instrument_type=lit_(structure_type),
            value=sum_("component_value"),
            currency=first_(IV.currency),
        )
    )
    return cast(LazyFrame[InstrumentValue], fold.compute(contributions))
