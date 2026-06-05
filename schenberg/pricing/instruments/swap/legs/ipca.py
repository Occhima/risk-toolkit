"""IPCA / CPI inflation-linked swap leg.

cashflow = notional * (projected/base) * (1 + real_coupon * T) - notional.
IPCA and CPI share the same payoff and market; only the routed kind differs.
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
from schenberg.pricing.market import CURVES, FIXINGS, PROJECTED


class IpcaLegRequirements(MarketRequirements[SwapLegInput]):
    zero_rate: Term[float] = requires(CURVES.zero_rate())
    base_index: Term[float] = requires(FIXINGS.base_index())
    projected_index: Term[float] = requires(PROJECTED.projected_index())


def _build(name: str) -> PricingGraph:
    g = PricingGraph[SwapLegInput, IpcaLegRequirements, LegPricing](name)
    c, m = g.contract, g.market
    year_fraction = year_fraction_term(g, payment_days=c.payment_days)

    @g.formula(tags=("inflation",))
    def inflation_factor(
        base_index: pl.Expr = uses(m.base_index),
        projected_index: pl.Expr = uses(m.projected_index),
    ) -> pl.Expr:
        return projected_index / base_index

    @g.formula(tags=("coupon",))
    def real_coupon_factor(
        real_coupon: pl.Expr = uses(c.real_coupon), T: pl.Expr = uses(year_fraction)
    ) -> pl.Expr:
        return 1.0 + real_coupon * T

    @g.formula(tags=("cashflow",))
    def cashflow_amount(
        notional: pl.Expr = uses(c.notional),
        inflation_factor: pl.Expr = uses(inflation_factor),
        real_coupon_factor: pl.Expr = uses(real_coupon_factor),
    ) -> pl.Expr:
        return notional * inflation_factor * real_coupon_factor - notional

    return assemble_leg(
        g, zero_rate=m.zero_rate, cashflow_amount=cashflow_amount, year_fraction=year_fraction
    )


ipca_swap_leg_graph = register(SwapLegKind.IPCA.value, graph=_build("ipca_swap_leg"))
cpi_swap_leg_graph = register("CPI", graph=_build("cpi_swap_leg"))
