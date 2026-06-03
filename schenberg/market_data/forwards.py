from __future__ import annotations

from dataclasses import dataclass

from schenberg.core.market import MarketRequirement
from schenberg.market_data.requirements import require


@dataclass(frozen=True, slots=True)
class EnergyForwardCurveSpec:
    name: str = "energy_forward_curve"

    def forward_price(
        self,
        *,
        submarket_col: str = "submarket",
        period_col: str = "delivery_period",
        price_output: str = "forward_price",
        settle_days_output: str = "payment_days",
    ) -> MarketRequirement:
        return require(
            self.name,
            (submarket_col, "submarket"),
            (period_col, "delivery_period"),
            outputs={
                "forward_price": price_output,
                "settle_days": settle_days_output,
            },
        )
