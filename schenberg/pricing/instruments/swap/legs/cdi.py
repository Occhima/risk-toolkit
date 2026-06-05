"""CDI swap leg: cashflow = notional * forward_rate * accrual.

CDI projects off the ``forward_rate`` carried on the curve, then takes the same
notional * rate * accrual shape as a fixed leg.
"""

from __future__ import annotations

import polars as pl

from schenberg.core.graph import PricingGraph, Term, uses
from schenberg.domain.enums import SwapLegKind
from schenberg.domain.schemas import LegPricing, SwapLegInput
from schenberg.market_data.requirements import MarketRequirements, requires
from schenberg.pricing.discounting import year_fraction_term
from schenberg.pricing.instruments.swap.generic import assemble_leg
from schenberg.pricing.instruments.swap.legs.registry import register
from schenberg.pricing.market import CURVES


class CdiLegRequirements(MarketRequirements[SwapLegInput]):
    zero_rate: Term[float] = requires(CURVES.zero_rate())
    forward_rate: Term[float] = requires(CURVES.forward_rate())


def _build() -> PricingGraph:
    g = PricingGraph[SwapLegInput, CdiLegRequirements, LegPricing]("cdi_swap_leg")
    c, m = g.contract, g.market
    year_fraction = year_fraction_term(g, payment_days=c.payment_days)

    @g.formula(tags=("cashflow", "cdi"))
    def cashflow_amount(
        notional: pl.Expr = uses(c.notional),
        forward_rate: pl.Expr = uses(m.forward_rate),
        accrual: pl.Expr = uses(c.accrual),
    ) -> pl.Expr:
        return notional * forward_rate * accrual

    return assemble_leg(
        g, zero_rate=m.zero_rate, cashflow_amount=cashflow_amount, year_fraction=year_fraction
    )


cdi_swap_leg_graph = register(SwapLegKind.CDI.value, graph=_build())
