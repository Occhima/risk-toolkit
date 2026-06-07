"""DI futures pricing graph and DV01 risk simulation.

Run with `uv run marimo edit docs/examples/12_di_futures_risk.py`.
"""

from __future__ import annotations

from datetime import date

import marimo
import polars as pl

from schenberg import FormulaGraph, SchenbergDataFrameModel

__generated_with = "0.13.15"
app = marimo.App(width="medium")


class DIFutureInput(SchenbergDataFrameModel):
    instrument_id: str
    maturity_code: str
    pricing_date: date
    maturity_date: date
    business_days: int
    entry_pu: float
    contracts: float
    contract_multiplier: float
    side_multiplier: float
    forward_rate: float


di_future_graph = FormulaGraph("b3_di_future", input=DIFutureInput)


@di_future_graph.formula(symbol="T")
def year_fraction(c):
    return c.business_days / 252.0


@di_future_graph.formula(symbol="PU")
def pu(c, year_fraction):
    return 100_000.0 / ((1.0 + c.forward_rate) ** year_fraction)


@di_future_graph.formula(symbol="PU_{+1bp}")
def pu_up_1bp(c, year_fraction):
    return 100_000.0 / ((1.0 + c.forward_rate + 0.0001) ** year_fraction)


@di_future_graph.formula(symbol="PnL")
def value(c, pu):
    return c.side_multiplier * c.contracts * c.contract_multiplier * (pu - c.entry_pu)


@di_future_graph.formula(symbol="DV01")
def dv01_1bp(c, pu, pu_up_1bp):
    return c.side_multiplier * c.contracts * c.contract_multiplier * (pu_up_1bp - pu)


di_future_graph.returns(
    "output",
    instrument_id="instrument_id",
    maturity_code="maturity_code",
    business_days="business_days",
    forward_rate="forward_rate",
    pu="pu",
    value="value",
    dv01_1bp="dv01_1bp",
)


def sample_b3_di_curve() -> pl.DataFrame:
    """Small B3-style DI futures curve snapshot for offline examples.

    Replace this frame with a real `brasa` dataset or a Banco Central do Brasil
    PRE curve feed in production; the rest of the notebook stays unchanged.
    """
    return pl.DataFrame(
        {
            "instrument_id": ["DI1F27", "DI1F28", "DI1F29", "DI1F31"],
            "maturity_code": ["F27", "F28", "F29", "F31"],
            "pricing_date": [date(2026, 6, 5)] * 4,
            "maturity_date": [
                date(2027, 1, 4),
                date(2028, 1, 3),
                date(2029, 1, 2),
                date(2031, 1, 2),
            ],
            "business_days": [145, 397, 648, 1150],
            "entry_pu": [94_025.0, 84_940.0, 76_380.0, 58_820.0],
            "contracts": [250.0, -180.0, 120.0, -75.0],
            "contract_multiplier": [1.0] * 4,
            "side_multiplier": [1.0, 1.0, 1.0, 1.0],
            "forward_rate": [0.1138, 0.1215, 0.1268, 0.1322],
        }
    )


def shock_curve(curve: pl.DataFrame) -> pl.DataFrame:
    return curve.with_columns(
        [
            (pl.col("forward_rate") + 0.0001).alias("parallel_up_1bp"),
            (pl.col("forward_rate") + 0.0025).alias("parallel_up_25bp"),
            (pl.col("forward_rate") - 0.0025).alias("parallel_down_25bp"),
            (
                pl.col("forward_rate")
                + pl.when(pl.col("business_days") <= 252).then(0.0010).otherwise(-0.0015)
            ).alias("steepener_short_up_long_down"),
        ]
    )


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md("""
    # Simulação de risco em futuros DI B3

    O grafo precifica PU de DI futuro com `forward_rate` **dentro** do grafo e já
    devolve `DV01` por vértice. A visualização não depende de Markdown renderizar
    Mermaid/LaTeX: grafo e fórmulas são PNGs por endpoint.
    """)
    return mo


@app.cell
def _(mo):
    import importlib

    import polars as pl
    from schenberg_viz import graph_png_url, latex_png_url

    risk = importlib.import_module("docs.examples.12_di_futures_risk")
    curve = risk.sample_b3_di_curve()
    priced = risk.di_future_graph.plan(curve.lazy(), view="output").collect()
    shocked = risk.shock_curve(curve)
    scenario_rows = []
    for scenario in [
        "forward_rate",
        "parallel_up_25bp",
        "parallel_down_25bp",
        "steepener_short_up_long_down",
    ]:
        scenario_frame = shocked.with_columns(pl.col(scenario).alias("forward_rate"))
        scenario_price = risk.di_future_graph.plan(scenario_frame.lazy(), view="output").collect()
        scenario_rows.append(
            scenario_price.select(
                pl.lit(scenario).alias("scenario"),
                "instrument_id",
                "maturity_code",
                "forward_rate",
                "pu",
                "value",
                "dv01_1bp",
            )
        )
    scenarios = pl.concat(scenario_rows)
    risk_by_scenario = scenarios.group_by("scenario").agg(
        pl.sum("value").alias("portfolio_value"),
        pl.sum("dv01_1bp").alias("portfolio_dv01_1bp"),
    )
    graph_url = graph_png_url(risk.di_future_graph, math_labels=True, view="output")
    formula_cards = [
        mo.vstack([mo.md(f"**{name}**"), mo.image(latex_png_url(formula), alt=formula)])
        for name, formula in risk.di_future_graph.formulas().items()
    ]
    return curve, formula_cards, graph_url, mo, priced, risk_by_scenario, scenarios, shocked


@app.cell(hide_code=True)
def _(curve, formula_cards, graph_url, mo, priced, risk_by_scenario, scenarios):
    mo.vstack(
        [
            mo.md("## Curva/boleta DI usada no exemplo"),
            mo.ui.table(curve),
            mo.md("## Grafo de pricing DI com forward rate (PNG)"),
            mo.image(graph_url, alt="B3 DI future pricing graph", width="100%"),
            mo.md("## Fórmulas renderizadas como PNG"),
            mo.vstack(formula_cards),
            mo.md("## Preço e DV01 por vencimento"),
            mo.ui.table(priced),
            mo.md("## Cenários por vencimento"),
            mo.ui.table(scenarios),
            mo.md("## Risco consolidado"),
            mo.ui.table(risk_by_scenario),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
