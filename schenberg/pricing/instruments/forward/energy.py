"""Energy forward pricing built on the generic forward valuation backbone."""

from __future__ import annotations

from typing import cast

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.core.graph import ExprGraph
from schenberg.core.market import MarketSnapshot, curve, energy_forward, fx
from schenberg.domain.schemas import EnergyForward, EnergyForwardOutput, ForwardPricing
from schenberg.pricing.instruments.forward.generic import forward_valuation_graph

energy_cashflow_graph = ExprGraph("energy_forward_cashflow")


@energy_cashflow_graph.node(tags=("energy", "cashflow"))
def future_value(
    quantity: pl.Expr,
    forward_price: pl.Expr,
    strike: pl.Expr,
    pay_receive: pl.Expr,
) -> pl.Expr:
    return pay_receive * quantity * (forward_price - strike)


energy_forward_graph = (
    ExprGraph.compose("energy_forward", forward_valuation_graph, energy_cashflow_graph)
    .with_market(
        energy_forward(),  # supplies forward_price and payment_days from the delivery-period curve
        curve("zero_rate"),
        fx(),
    )
    .with_outputs("pricing", ForwardPricing)
)


def explode_delivery(contracts: pl.LazyFrame) -> pl.LazyFrame:
    """Explode a block contract into one row per delivery period."""
    return (
        contracts.explode("delivery_periods")
        .rename({"delivery_periods": "delivery_period"})
        .with_columns(
            pl.col("delivery_period").cast(pl.Utf8),
            pay_receive=pl.when(pl.col("buy_sell") == "BUY").then(1.0).otherwise(-1.0),
        )
    )


@pa.check_types(lazy=True)
def price_energy_forward(
    contracts: LazyFrame[EnergyForward],
    market: MarketSnapshot,
) -> LazyFrame[EnergyForwardOutput]:
    """Price energy forwards and aggregate delivery-period rows by contract."""
    legs = explode_delivery(contracts)
    priced = energy_forward_graph.compute_for(legs, market=market, output_profile="pricing")
    result = priced.group_by("contract_id").agg(
        mtm_local=pl.col("present_value").sum(),
        mtm=pl.col("value").sum(),
    )
    return cast(LazyFrame[EnergyForwardOutput], result)
