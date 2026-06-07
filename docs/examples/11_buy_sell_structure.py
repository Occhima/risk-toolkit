"""Buy/sell structure example with rendered graph and formula images.

Run with `uv run marimo edit docs/examples/11_buy_sell_structure.py`.
"""

from __future__ import annotations

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md("""
    # Estrutura compra/venda

    Exemplo de **call spread**: compra uma call ATM e vende uma call OTM. A célula não
    depende de Mermaid ou LaTeX em Markdown cru: o grafo vem como PNG e as fórmulas
    também são imagens geradas por endpoint.
    """)
    return mo


@app.cell
def _(mo):
    import importlib

    import polars as pl
    from schenberg.structure import structure_value
    from schenberg_viz import graph_png_url, latex_png_url

    vanilla = importlib.import_module("docs.examples.04_vanilla_option")
    trades = pl.DataFrame(
        {
            "instrument_id": ["BUY-CALL-100", "SELL-CALL-110"],
            "option_type": ["call", "call"],
            "currency_pair": ["USD/BRL", "USD/BRL"],
            "curve": ["BRL_DI", "BRL_DI"],
            "currency": ["BRL", "BRL"],
            "pricing_date": [vanilla.date(2026, 6, 6), vanilla.date(2026, 6, 6)],
            "expiry": [vanilla.date(2027, 6, 6), vanilla.date(2027, 6, 6)],
            "strike": [100.0, 110.0],
            "time_to_maturity": [1.0, 1.0],
            "payment_days": [252, 252],
        }
    ).lazy()
    market = vanilla.sample_market(spot=103.0, rate=0.1175, vol=0.215)
    priced = vanilla.price_vanilla_option(trades, market).with_columns(
        pl.lit("OPTION").alias("instrument_type")
    )
    legs = pl.DataFrame(
        {
            "structure_id": ["CALL-SPREAD-001", "CALL-SPREAD-001"],
            "leg_id": ["long_atm_call", "short_otm_call"],
            "instrument_type": ["OPTION", "OPTION"],
            "instrument_id": ["BUY-CALL-100", "SELL-CALL-110"],
            "quantity": [1_000.0, 1_000.0],
            "weight": [1.0, -1.0],
            "side": ["BUY", "SELL"],
        }
    ).lazy()
    structure = structure_value(legs, priced)
    graph_url = graph_png_url(vanilla.vanilla_option_graph, math_labels=True)
    formula_cards = [
        mo.vstack([mo.md(f"**{name}**"), mo.image(latex_png_url(formula), alt=formula)])
        for name, formula in vanilla.vanilla_option_graph.formulas().items()
    ]
    return formula_cards, graph_url, legs, market, mo, priced, structure, trades, vanilla


@app.cell(hide_code=True)
def _(formula_cards, graph_url, legs, mo, priced, structure):
    legs_view = legs.collect()
    priced_view = priced.select(
        "instrument_id", "option_type", "strike", "value", "delta", "gamma", "vega"
    ).collect()
    structure_view = structure.collect()
    mo.vstack(
        [
            mo.md("## Grafo de pricing das legs (PNG)"),
            mo.image(graph_url, alt="Option graph", width="100%"),
            mo.md("## Fórmulas das legs como PNG"),
            mo.vstack(formula_cards),
            mo.md("## Boleta da estrutura"),
            mo.ui.table(legs_view),
            mo.md("## Preço das legs"),
            mo.ui.table(priced_view),
            mo.md("## Valor consolidado da estrutura"),
            mo.ui.table(structure_view),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
