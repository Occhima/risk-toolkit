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


# Legacy requirement helpers retained for swap compatibility. New market-data
# specs live under schenberg.market_data.* and should be preferred for pricing.
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


def energy_forward(
    *,
    submarket_col: str = "submarket",
    period_col: str = "delivery_period",
    outputs: dict[str, str] | None = None,
) -> MarketRequirement:
    out = outputs or {"forward_price": "forward_price", "settle_days": "payment_days"}
    return MarketRequirement(
        table="forward_curves",
        on=ColumnSet.from_pairs(
            (submarket_col, "submarket"),
            (period_col, "delivery_period"),
        ),
        outputs=out,
    )


def fx(*, currency_col: str = "currency", output: str = "fx_rate") -> MarketRequirement:
    return MarketRequirement(
        table="fx_rates",
        on=ColumnSet.from_pairs((currency_col, "currency")),
        outputs={"fx_rate": output},
    )
