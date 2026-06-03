from __future__ import annotations

from schenberg.core.columns import ColumnSet
from schenberg.core.market import MarketRequirement


def require(
    table: str,
    *pairs: tuple[str, str],
    outputs: dict[str, str],
    optional: bool = False,
) -> MarketRequirement:
    return MarketRequirement(
        table=table,
        on=ColumnSet.from_pairs(*pairs),
        outputs=outputs,
        optional=optional,
    )
