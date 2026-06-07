"""Forward simples com FormulaGraph decorado e delta.

Execute with `uv run poe examples-html`.
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Forward simples — grafo decorado, market bind e delta

        Este exemplo é autocontido: o pricer específico é declarado aqui com a
        API pública do Schenberg. O grafo é simbólico, preguiçoso em Polars e
        inspecionável por `explain`, Mermaid, LaTeX e `stage`.
        """
    )
    return


@app.cell
def _():
    from datetime import date

    import marimo as mo
    import polars as pl
    from schenberg import FormulaGraph, MarketSnapshot, With, bind, exp, market_role
    from schenberg.domain.base import SchenbergDataFrameModel

    return (
        FormulaGraph,
        MarketSnapshot,
        SchenbergDataFrameModel,
        With,
        bind,
        date,
        exp,
        market_role,
        mo,
        pl,
    )


@app.cell
def _(SchenbergDataFrameModel, With, market_role):
    ForwardRate = (
        market_role("forward_rate")
        .read("curves", "forward_rate")
        .by(indexer="id_indexador", payment_days="tenor_days")
    )
    RiskFreeRate = (
        market_role("risk_free_rate")
        .read("curves", "risk_free_rate")
        .by(indexer="id_indexador", payment_days="tenor_days")
    )

    class ForwardInput(With[ForwardRate], With[RiskFreeRate], SchenbergDataFrameModel):
        instrument_id: str
        indexer: str
        currency: str
        strike: float
        payment_days: int

    return ForwardInput


@app.cell
def _(FormulaGraph, ForwardInput, exp):
    graph = FormulaGraph("forward", input=ForwardInput)

    @graph.formula(symbol="T")
    def year_fraction(c):
        return c.payment_days / 252.0

    @graph.formula(symbol="DF")
    def discount_factor(c, year_fraction):
        return exp(-c.risk_free_rate * year_fraction)

    @graph.formula(symbol="FV")
    def future_value(c):
        return c.forward_rate - c.strike

    @graph.formula(symbol="PV")
    def present_value(future_value, discount_factor):
        return future_value * discount_factor

    @graph.formula(symbol="Delta")
    def delta(discount_factor):
        return discount_factor

    graph.returns(
        "output",
        instrument_id="instrument_id",
        future_value="future_value",
        present_value="present_value",
        value="present_value",
        delta="delta",
        currency="currency",
    )
    return graph


@app.cell(hide_code=True)
def _(graph, mo):
    mo.vstack(
        [
            mo.md("## Introspecção"),
            mo.md(
                "### `graph.explain(view='output')`\n```text\n"
                + graph.explain(view="output")
                + "\n```"
            ),
            mo.md("### Mermaid\n```text\n" + graph.to_mermaid(math_labels=True) + "\n```"),
            mo.md("### Fórmulas\n```text\n" + "\n".join(graph.formulas().values()) + "\n```"),
        ]
    )
    return


@app.cell
def _(ForwardInput, MarketSnapshot, bind, date, graph, pl):
    trades = pl.DataFrame(
        {
            "instrument_id": ["FWD-SOY-1", "FWD-SOY-2", "FWD-CORN-1"],
            "indexer": ["SOY", "SOY", "CORN"],
            "currency": ["USD", "USD", "USD"],
            "strike": [1000.0, 1040.0, 550.0],
            "payment_days": [252, 504, 252],
        }
    ).lazy()
    market = (
        MarketSnapshot.at(date(2026, 6, 6))
        .source(
            "curves",
            pl.DataFrame(
                {
                    "id_indexador": ["SOY", "SOY", "CORN"],
                    "tenor_days": [252, 504, 252],
                    "forward_rate": [1050.0, 1120.0, 575.0],
                    "risk_free_rate": [0.10, 0.12, 0.09],
                }
            ),
            unique_by=("id_indexador", "tenor_days"),
        )
        .build()
    )
    enriched = bind(trades, market, ForwardInput)
    lazy_priced = graph.plan(enriched, view="output")
    priced = lazy_priced.collect()
    return lazy_priced, priced


@app.cell(hide_code=True)
def _(lazy_priced, mo, priced):
    mo.vstack(
        [
            mo.md("## Tabela precificada"),
            mo.ui.table(priced),
            mo.md("## Plano lazy até a exibição\n```text\n" + lazy_priced.explain() + "\n```"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
