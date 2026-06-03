"""Shared swap-leg valuation primitives.

A swap leg is the shared :data:`~schenberg.pricing.discounting.discount_graph`
backbone plus a direction (pay/receive), a discounted signed cashflow, and an FX
translation into the reporting currency — the same "discount a cashflow, then
convert" tail a forward uses, differing only in the payoff a leg feeds in (see
:mod:`.legs`). ``fx_rate`` is optional market data: a leg in a foreign currency
declares the FX source and carries a ``currency``; otherwise the rate defaults to
1.0 and the leg stays in its local currency, exactly as before.
"""

from __future__ import annotations

import polars as pl

from schenberg.core.graph import ExprGraph
from schenberg.core.market import curve
from schenberg.domain.enums import PayReceive
from schenberg.domain.schemas import LegPricing
from schenberg.market_data.fx import FxRatesSpec
from schenberg.pricing.discounting import discount_graph

FX = FxRatesSpec("fx_rates")

_leg_payoff = ExprGraph("swap_leg_payoff")


@_leg_payoff.node(dtype=pl.Float64, tags=("direction",))
def pay_receive_sign(pay_receive: pl.Expr) -> pl.Expr:
    return pl.when(pay_receive == PayReceive.RECEIVE.value).then(1.0).otherwise(-1.0)


@_leg_payoff.node(dtype=pl.Float64, tags=("cashflow",))
def signed_cashflow(cashflow_amount: pl.Expr, pay_receive_sign: pl.Expr) -> pl.Expr:
    return cashflow_amount * pay_receive_sign


@_leg_payoff.node(dtype=pl.Float64, tags=("pricing",))
def local_pv(signed_cashflow: pl.Expr, discount_factor: pl.Expr) -> pl.Expr:
    return signed_cashflow * discount_factor


@_leg_payoff.node(dtype=pl.Float64, tags=("pricing", "fx"))
def pv(local_pv: pl.Expr, fx_rate: pl.Expr) -> pl.Expr:
    # fx_rate is optional market data; absent/null means the reporting currency
    # equals the leg currency.
    return local_pv * fx_rate.fill_null(1.0)


swap_leg_valuation_graph = ExprGraph.compose(
    "swap_leg_valuation", discount_graph, _leg_payoff
).with_outputs("pricing", LegPricing)

base_swap_leg_graph = (
    ExprGraph.compose("base_swap_leg", swap_leg_valuation_graph)
    .with_market(curve("zero_rate"), FX.fx_rate(optional=True))
    .with_outputs("pricing", LegPricing)
)
