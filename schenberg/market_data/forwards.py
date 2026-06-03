from __future__ import annotations

from dataclasses import dataclass

from schenberg.core.columns import ColumnLike, ColumnSet, col_name
from schenberg.core.market import MarketRequirement


@dataclass(frozen=True, slots=True)
class EnergyForwardCurveSpec:
    name: str = "energy_forward_curve"

    def forward_price(
        self,
        *,
        submarket: ColumnLike = "submarket",
        period: ColumnLike = "delivery_period",
        price_output: str = "forward_price",
        settle_days_output: str = "payment_days",
    ) -> MarketRequirement:
        return MarketRequirement(
            table=self.name,
            on=ColumnSet.from_pairs(
                (col_name(submarket), "submarket"),
                (col_name(period), "delivery_period"),
            ),
            outputs={
                "forward_price": price_output,
                "settle_days": settle_days_output,
            },
        )
