"""Small lazy option PnL explain by sequential repricing."""

from __future__ import annotations

import marimo
import polars as pl

import importlib

vanilla = importlib.import_module("docs.examples.04_vanilla_option")

__generated_with = "0.13.15"
app = marimo.App(width="medium")


def option_pnl_explain(
    trades: pl.LazyFrame,
    market_d0,
    market_spot,
    market_vol,
    market_rate,
    market_d1,
) -> pl.LazyFrame:
    """Explain D0->D1 as spot, vol, rate, and residual lazy repricing buckets."""
    ids = ["instrument_id", "currency"]
    base = vanilla.price_vanilla_option(trades, market_d0).select(
        *ids, pl.col("value").alias("base_value")
    )
    spot = vanilla.price_vanilla_option(trades, market_spot).select(
        "instrument_id", pl.col("value").alias("spot_step_value")
    )
    vol = vanilla.price_vanilla_option(trades, market_vol).select(
        "instrument_id", pl.col("value").alias("vol_step_value")
    )
    rate = vanilla.price_vanilla_option(trades, market_rate).select(
        "instrument_id", pl.col("value").alias("rate_step_value")
    )
    new = vanilla.price_vanilla_option(trades, market_d1).select(
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
    trades = vanilla.sample_trades()
    d0 = vanilla.sample_market(spot=100.0, rate=0.05, vol=0.20)
    spot = vanilla.sample_market(spot=101.0, rate=0.05, vol=0.20)
    vol = vanilla.sample_market(spot=101.0, rate=0.05, vol=0.21)
    rate = vanilla.sample_market(spot=101.0, rate=0.051, vol=0.21)
    d1 = vanilla.sample_market(spot=101.0, rate=0.051, vol=0.21)
    return option_pnl_explain(trades, d0, spot, vol, rate, d1)


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md("""
    # Option PnL explain

    This is a compact lazy repricing explain: D0, spot step, vol step, rate step,
    D1, and residual. The option graph remains pure and has no position direction.
    """)
    return mo


@app.cell
def _(mo):
    import importlib

    pnl = importlib.import_module("docs.examples.05_option_pnl_explain")
    explain = pnl.sample_pnl_explain()
    return explain, mo


@app.cell(hide_code=True)
def _(explain, mo):
    mo.ui.table(explain.collect())
    return


if __name__ == "__main__":
    app.run()
