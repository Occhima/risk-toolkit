"""Fixed-rate swap leg: cashflow = notional * fixed_rate * accrual."""

from __future__ import annotations

import polars as pl

from schenberg.core.graph import PricingGraph, uses
from schenberg.domain.enums import SwapLegKind
from schenberg.domain.schemas import LegPricing, SwapLegInput
from schenberg.pricing.discounting import year_fraction_term
from schenberg.pricing.instruments.swap.generic import DiscountRequirements, assemble_leg
from schenberg.pricing.instruments.swap.legs.registry import register


def _build() -> PricingGraph:
    g = PricingGraph[SwapLegInput, DiscountRequirements, LegPricing]("fixed_swap_leg")
    c, m = g.contract, g.market
    year_fraction = year_fraction_term(g, payment_days=c.payment_days)

    @g.formula(tags=("fixed", "cashflow"))
    def cashflow_amount(
        notional: pl.Expr = uses(c.notional),
        fixed_rate: pl.Expr = uses(c.fixed_rate),
        accrual: pl.Expr = uses(c.accrual),
    ) -> pl.Expr:
        return notional * fixed_rate * accrual

    return assemble_leg(
        g, zero_rate=m.zero_rate, cashflow_amount=cashflow_amount, year_fraction=year_fraction
    )


fixed_swap_leg_graph = register(SwapLegKind.FIXED.value, graph=_build())
