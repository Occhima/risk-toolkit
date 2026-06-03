"""Declarative market requirements."""

from __future__ import annotations

from dataclasses import dataclass

from schenberg.core.columns import ColumnSet


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
