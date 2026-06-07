"""Shared vanilla-option pricer for the integration tests.

Instrument formulas live *outside* the core library: this module assembles a
Black-Scholes pricing graph from the public Schenberg API using headless formula
parameters, exactly as the Quarto example notebooks do. Tests import it instead
of reaching into the documentation notebooks.
"""

from __future__ import annotations

from datetime import date

import polars as pl
from pandera.typing.polars import LazyFrame
from schenberg import (
    CURVES,
    FIXINGS,
    VOLS,
    FormulaGraph,
    MarketSnapshot,
    SchenbergDataFrameModel,
    With,
    bind,
    exp,
    log,
    normal_cdf,
    normal_pdf,
    price_function,
    sqrt,
    where,
)

RiskFree = (
    CURVES.zero_rate("BRL_DI", as_="risk_free_rate").source("curves").for_tenor("payment_days")
)
Spot = FIXINGS.value(as_="spot").source("fixings").by(currency_pair="currency_pair")
Vol = (
    VOLS.implied("USD/BRL", as_="vol")
    .source("vol_surface")
    .for_expiry("expiry")
    .for_strike("strike")
)


class VanillaOptionTrade(SchenbergDataFrameModel):
    instrument_id: str
    option_type: str
    currency_pair: str
    curve: str
    currency: str
    pricing_date: date
    expiry: date
    strike: float
    time_to_maturity: float
    payment_days: int


class VanillaOptionInput(
    With[Spot],
    With[RiskFree],
    With[Vol],
    SchenbergDataFrameModel,
):
    instrument_id: str
    option_type: str
    currency_pair: str
    curve: str
    currency: str
    pricing_date: date
    expiry: date
    strike: float
    time_to_maturity: float
    payment_days: int


vanilla_option_graph = FormulaGraph("vanilla_option", input=VanillaOptionInput)


@vanilla_option_graph.formula(symbol="d_1")
def d1(spot, strike, risk_free_rate, vol, time_to_maturity):
    return (log(spot / strike) + (risk_free_rate + 0.5 * vol**2) * time_to_maturity) / (
        vol * sqrt(time_to_maturity)
    )


@vanilla_option_graph.formula(symbol="d_2")
def d2(d1, vol, time_to_maturity):
    return d1 - vol * sqrt(time_to_maturity)


@vanilla_option_graph.formula(symbol="C")
def call_value(spot, strike, risk_free_rate, time_to_maturity, d1, d2):
    return spot * normal_cdf(d1) - strike * exp(-risk_free_rate * time_to_maturity) * normal_cdf(d2)


@vanilla_option_graph.formula(symbol="P")
def put_value(spot, strike, risk_free_rate, time_to_maturity, d1, d2):
    return strike * exp(-risk_free_rate * time_to_maturity) * normal_cdf(-d2) - spot * normal_cdf(
        -d1
    )


@vanilla_option_graph.formula(symbol="V")
def value(option_type, call_value, put_value):
    return where(option_type == "call", call_value, put_value)


@vanilla_option_graph.formula(symbol="\\Delta")
def delta(option_type, d1):
    return where(option_type == "call", normal_cdf(d1), normal_cdf(d1) - 1.0)


@vanilla_option_graph.formula(symbol="\\Gamma")
def gamma(spot, vol, time_to_maturity, d1):
    return normal_pdf(d1) / (spot * vol * sqrt(time_to_maturity))


@vanilla_option_graph.formula(symbol="Vega")
def vega(spot, time_to_maturity, d1):
    return spot * normal_pdf(d1) * sqrt(time_to_maturity)


vanilla_option_graph.returns(
    "output",
    instrument_type="instrument_type",
    instrument_id="instrument_id",
    option_type="option_type",
    currency="currency",
    spot="spot",
    strike="strike",
    risk_free_rate="risk_free_rate",
    vol="vol",
    time_to_maturity="time_to_maturity",
    d1="d1",
    d2="d2",
    call_value="call_value",
    put_value="put_value",
    value="value",
    delta="delta",
    gamma="gamma",
    vega="vega",
)


@price_function
def price_vanilla_option(
    trades: LazyFrame[VanillaOptionTrade],
    market: MarketSnapshot,
) -> pl.LazyFrame:
    enriched = bind(trades, market, VanillaOptionInput)
    return vanilla_option_graph.plan(
        enriched.with_columns(pl.lit("VANILLA_OPTION").alias("instrument_type")), view="output"
    )


def sample_trades() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "instrument_id": ["OPT-CALL-1", "OPT-PUT-1"],
            "option_type": ["call", "put"],
            "currency_pair": ["USD/BRL", "USD/BRL"],
            "curve": ["BRL_DI", "BRL_DI"],
            "currency": ["BRL", "BRL"],
            "pricing_date": [date(2026, 6, 6), date(2026, 6, 6)],
            "expiry": [date(2027, 6, 6), date(2027, 6, 6)],
            "strike": [100.0, 100.0],
            "time_to_maturity": [1.0, 1.0],
            "payment_days": [252, 252],
        }
    ).lazy()


def sample_market(*, spot: float = 100.0, rate: float = 0.05, vol: float = 0.20) -> MarketSnapshot:
    return (
        MarketSnapshot.at(date(2026, 6, 6))
        .source(
            "fixings",
            pl.DataFrame({"currency_pair": ["USD/BRL"], "value": [spot]}),
            unique_by=("currency_pair",),
        )
        .source(
            "curves",
            pl.DataFrame({"curve": ["BRL_DI"], "tenor_days": [252], "zero_rate": [rate]}),
            unique_by=("curve", "tenor_days"),
        )
        .source(
            "vol_surface",
            pl.DataFrame(
                {
                    "currency_pair": ["USD/BRL"],
                    "expiry": [date(2027, 6, 6)],
                    "strike": [100.0],
                    "implied_vol": [vol],
                }
            ),
            unique_by=("currency_pair", "expiry", "strike"),
        )
        .build()
    )


def option_pnl_explain(
    trades: pl.LazyFrame,
    market_d0: MarketSnapshot,
    market_spot: MarketSnapshot,
    market_vol: MarketSnapshot,
    market_rate: MarketSnapshot,
    market_d1: MarketSnapshot,
) -> pl.LazyFrame:
    """Explain D0->D1 as spot, vol, rate, and residual lazy repricing buckets."""
    ids = ["instrument_id", "currency"]
    base = price_vanilla_option(trades, market_d0).select(*ids, pl.col("value").alias("base_value"))
    spot = price_vanilla_option(trades, market_spot).select(
        "instrument_id", pl.col("value").alias("spot_step_value")
    )
    vol = price_vanilla_option(trades, market_vol).select(
        "instrument_id", pl.col("value").alias("vol_step_value")
    )
    rate = price_vanilla_option(trades, market_rate).select(
        "instrument_id", pl.col("value").alias("rate_step_value")
    )
    new = price_vanilla_option(trades, market_d1).select(
        "instrument_id", pl.col("value").alias("new_value")
    )
    return (
        base.join(spot, on="instrument_id", how="left")
        .join(vol, on="instrument_id", how="left")
        .join(rate, on="instrument_id", how="left")
        .join(new, on="instrument_id", how="left")
        .with_columns(
            (pl.col("spot_step_value") - pl.col("base_value")).alias("spot_value_pnl"),
            (pl.col("vol_step_value") - pl.col("spot_step_value")).alias("vol_value_pnl"),
            (pl.col("rate_step_value") - pl.col("vol_step_value")).alias("rate_value_pnl"),
            (pl.col("new_value") - pl.col("base_value")).alias("total_value_pnl"),
        )
        .with_columns(
            (
                pl.col("total_value_pnl")
                - pl.col("spot_value_pnl")
                - pl.col("vol_value_pnl")
                - pl.col("rate_value_pnl")
            ).alias("residual_value_pnl")
        )
        .select(
            "instrument_id",
            "currency",
            "base_value",
            "new_value",
            "spot_value_pnl",
            "vol_value_pnl",
            "rate_value_pnl",
            "residual_value_pnl",
            "total_value_pnl",
        )
    )


def sample_pnl_explain() -> pl.LazyFrame:
    trades = sample_trades()
    d0 = sample_market(spot=100.0, rate=0.05, vol=0.20)
    spot = sample_market(spot=101.0, rate=0.05, vol=0.20)
    vol = sample_market(spot=101.0, rate=0.05, vol=0.21)
    rate = sample_market(spot=101.0, rate=0.051, vol=0.21)
    d1 = sample_market(spot=101.0, rate=0.051, vol=0.21)
    return option_pnl_explain(trades, d0, spot, vol, rate, d1)
