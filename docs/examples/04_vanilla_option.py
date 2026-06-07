"""Vanilla European option with semantic market roles and Black-Scholes IR.

Execute with `uv run poe examples-vanilla-option-html`.
"""

from __future__ import annotations

from datetime import date

import marimo
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

__generated_with = "0.13.15"
app = marimo.App(width="medium")

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
def d1(c):
    return (log(c.spot / c.strike) + (c.risk_free_rate + 0.5 * c.vol**2) * c.time_to_maturity) / (
        c.vol * sqrt(c.time_to_maturity)
    )


@vanilla_option_graph.formula(symbol="d_2")
def d2(c, d1):
    return d1 - c.vol * sqrt(c.time_to_maturity)


@vanilla_option_graph.formula(symbol="C")
def call_value(c, d1, d2):
    return c.spot * normal_cdf(d1) - c.strike * exp(
        -c.risk_free_rate * c.time_to_maturity
    ) * normal_cdf(d2)


@vanilla_option_graph.formula(symbol="P")
def put_value(c, d1, d2):
    return c.strike * exp(-c.risk_free_rate * c.time_to_maturity) * normal_cdf(
        -d2
    ) - c.spot * normal_cdf(-d1)


@vanilla_option_graph.formula(symbol="V")
def value(c, call_value, put_value):
    return where(c.option_type == "call", call_value, put_value)


@vanilla_option_graph.formula(symbol="\\Delta")
def delta(c, d1):
    return where(c.option_type == "call", normal_cdf(d1), normal_cdf(d1) - 1.0)


@vanilla_option_graph.formula(symbol="\\Gamma")
def gamma(c, d1):
    return normal_pdf(d1) / (c.spot * c.vol * sqrt(c.time_to_maturity))


@vanilla_option_graph.formula(symbol="Vega")
def vega(c, d1):
    return c.spot * normal_pdf(d1) * sqrt(c.time_to_maturity)


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


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md("""
    # Vanilla Black-Scholes option

    Market data is declared with semantic roles outside the graph. `bind(...)`
    resolves `spot`, `risk_free_rate`, and `vol`; the graph sees only those columns.
    """)
    return mo


@app.cell
def _(mo):
    import importlib

    from schenberg_viz import graph_png_url, latex_png_url

    vanilla = importlib.import_module("docs.examples.04_vanilla_option")
    trades = vanilla.sample_trades()
    market = vanilla.sample_market()
    enriched = vanilla.bind(trades, market, vanilla.VanillaOptionInput)
    priced = vanilla.price_vanilla_option(trades, market)
    staged = vanilla.vanilla_option_graph.stage(
        enriched.with_columns(vanilla.pl.lit("VANILLA_OPTION").alias("instrument_type")),
        view="output",
    )
    graph_url = graph_png_url(vanilla.vanilla_option_graph, math_labels=True)
    formula_cards = [
        mo.vstack([mo.md(f"**{name}**"), mo.image(latex_png_url(formula), alt=formula)])
        for name, formula in vanilla.vanilla_option_graph.formulas().items()
    ]
    return enriched, formula_cards, graph_url, market, mo, priced, staged, trades, vanilla


@app.cell(hide_code=True)
def _(formula_cards, graph_url, mo, priced, staged, vanilla):
    mo.vstack(
        [
            mo.md(
                "## `graph.explain(view='output')`\n```text\n"
                + vanilla.vanilla_option_graph.explain(view="output")
                + "\n```"
            ),
            mo.md("## Grafo de pricing renderizado (PNG)"),
            mo.image(graph_url, alt="Vanilla option pricing graph", width="100%"),
            mo.md(
                "## Mermaid fonte (debug)\n```text\n"
                + vanilla.vanilla_option_graph.to_mermaid(math_labels=True)
                + "\n```"
            ),
            mo.md("## Fórmulas renderizadas como PNG"),
            mo.vstack(formula_cards),
            mo.md("## Stage"),
            mo.ui.table(
                staged.select("instrument_id", "d1", "d2", "call_value", "put_value").collect()
            ),
            mo.md("## Price and Greeks"),
            mo.ui.table(
                priced.select(
                    "instrument_id", "option_type", "value", "delta", "gamma", "vega"
                ).collect()
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
