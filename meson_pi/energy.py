"""Energy forward pricing (BBCE-style power forwards).

Composes the GENERIC valuation backbone (discount + FX) and adds only the energy
cashflow. Market data arrives through a single MarketSnapshot via requirements,
exactly like swaps — no hand-rolled joins, no loose frames.
"""
from __future__ import annotations

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import LazyFrame

from .graph import ExprGraph
from .market import MarketSnapshot, curve, energy_forward, fx
from .swap.pricing_graph import discounting_graph     # year_fraction, discount_factor, leg_pv


# ---------------------------------------------------------------------------
# Generic valuation backbone (NOT energy-specific): future_value -> present_value
# (discounting_graph) -> value in reporting currency (fx_graph). Any instrument
# composes this and supplies only its own `signed_cashflow`.
# ---------------------------------------------------------------------------
fx_graph = ExprGraph("fx")


@fx_graph.node(tags=("fx",))
def pv_base(leg_pv, fx_rate):                  # present_value -> reporting currency
    return leg_pv * fx_rate


# discounting_graph + fx_graph = the reusable backbone.
valuation_backbone = ExprGraph.compose("valuation_backbone", discounting_graph, fx_graph)


# ---------------------------------------------------------------------------
# Boundary contracts
# ---------------------------------------------------------------------------
class EnergyForward(pa.DataFrameModel):
    """Block forward (wide). `delivery_periods` (List[str]) is intentionally not
    declared, so pandera doesn't coerce the list to object and break the explode."""
    contract_id: str
    submarket: str
    buy_sell: str
    id_indexador: int              # discount-curve id (the BRL discount curve)
    quantity: float
    strike: float
    currency: str


class EnergyForwardOutput(pa.DataFrameModel):
    contract_id: str
    mtm_local: float
    mtm: float


# ---------------------------------------------------------------------------
# Energy-specific cashflow + the leg graph (backbone + cashflow + market reqs)
# ---------------------------------------------------------------------------
energy_cashflow_graph = ExprGraph("energy_cashflow")


@energy_cashflow_graph.node(tags=("energy",))
def cashflow_amount(quantity, forward_price, strike):
    return quantity * (forward_price - strike)


@energy_cashflow_graph.node(tags=("energy",))
def signed_cashflow(pay_receive, cashflow_amount):
    return pay_receive * cashflow_amount


energy_forward_graph = (
    ExprGraph.compose("energy_forward", valuation_backbone, energy_cashflow_graph)
    # energy_forward FIRST: it sets payment_days (= settle_days) that curve then uses.
    .with_market(energy_forward(), curve("zero_rate"), fx())
    .with_outputs("pricing",
                  forward_price="forward_price",
                  cashflow_amount="cashflow_amount",
                  pv_local="leg_pv",     # present value, contract currency
                  pv="pv_base")          # present value, reporting currency
)


# ---------------------------------------------------------------------------
# Prepare: explode block contract -> one row per delivery period
# ---------------------------------------------------------------------------
def explode_delivery(contracts: pl.LazyFrame) -> pl.LazyFrame:
    return (
        contracts.explode("delivery_periods")
        .rename({"delivery_periods": "delivery_period"})
        .with_columns(
            pl.col("delivery_period").cast(pl.Utf8),
            pay_receive=pl.when(pl.col("buy_sell") == "BUY").then(1.0).otherwise(-1.0),
        )
    )


# ---------------------------------------------------------------------------
# Public API: ONE MarketSnapshot, like price_swap
# ---------------------------------------------------------------------------
@pa.check_types(lazy=True)
def price_energy_forward(contracts: LazyFrame[EnergyForward],
                         market: MarketSnapshot) -> LazyFrame[EnergyForwardOutput]:
    legs = explode_delivery(contracts)
    priced = energy_forward_graph.compute_for(legs, market=market, output_profile="pricing")
    return priced.group_by("contract_id").agg(
        mtm_local=pl.col("pv_local").sum(),
        mtm=pl.col("pv").sum(),
    )
