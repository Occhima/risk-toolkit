"""DF USD/BRL com PTAX no quinto dia útil antes do vencimento.

Execute with `uv run poe examples-html`.
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # DF USD/BRL com fixing customizado de PTAX

        A PTAX é lida no quinto dia útil antes do vencimento. O calendário aqui é
        deliberadamente simples (segunda a sexta) para manter o exemplo estável e
        versionado, sem chamadas online ao Banco Central.
        """
    )
    return


@app.cell
def _():
    from datetime import date

    import marimo as mo
    import polars as pl
    from schenberg import Fixing, FormulaGraph, MarketSnapshot, With, bind, exp, market_role
    from schenberg.domain.base import SchenbergDataFrameModel
    from schenberg.market_data.date_rules import previous_business_days

    return (
        Fixing,
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
        previous_business_days,
    )


@app.cell
def _(Fixing, SchenbergDataFrameModel, With, date, market_role, previous_business_days):
    ptax_fixing = Fixing.rule(previous_business_days("tenor", n=5))
    Ptax = (
        market_role("ptax")
        .read("ptax_fixings", "fixing_value")
        .by(currency_pair="currency_pair")
        .fixing("ptax_fixing_date", ptax_fixing)
    )
    RiskFree = (
        market_role("risk_free_rate").read("curves", "risk_free_rate").by(payment_days="tenor_days")
    )

    class UsdBrlDfInput(With[Ptax], With[RiskFree], SchenbergDataFrameModel):
        instrument_id: str
        currency_pair: str
        currency: str
        tenor: date
        ptax_fixing_date: date
        contracted_rate: float
        notional_usd: float
        payment_days: int

    return UsdBrlDfInput, ptax_fixing


@app.cell
def _(FormulaGraph, UsdBrlDfInput, exp):
    graph = FormulaGraph("usdbrl_df", input=UsdBrlDfInput)

    @graph.formula(symbol="T")
    def year_fraction(c):
        return c.payment_days / 252.0

    @graph.formula(symbol="DF")
    def discount_factor(c, year_fraction):
        return exp(-c.risk_free_rate * year_fraction)

    @graph.formula(symbol="PV")
    def present_value(c, discount_factor):
        return c.notional_usd * (c.contracted_rate - c.ptax) * discount_factor

    graph.returns(
        "output",
        instrument_id="instrument_id",
        ptax_fixing_date="ptax_fixing_date",
        ptax="ptax",
        value="present_value",
        currency="currency",
    )
    return graph


@app.cell
def _(MarketSnapshot, UsdBrlDfInput, bind, date, graph, pl, ptax_fixing):
    trades = (
        pl.DataFrame(
            {
                "instrument_id": ["NDF-USD-BRL-1"],
                "currency_pair": ["USD/BRL"],
                "currency": ["BRL"],
                "tenor": [date(2026, 6, 15)],
                "contracted_rate": [5.50],
                "notional_usd": [1_000_000.0],
                "payment_days": [21],
            }
        )
        .lazy()
        .with_columns(ptax_fixing.expr().alias("ptax_fixing_date"))
    )
    market = (
        MarketSnapshot.at(date(2026, 6, 6))
        .source(
            "ptax_fixings",
            pl.DataFrame(
                {
                    "currency_pair": ["USD/BRL", "USD/BRL"],
                    "ptax_fixing_date": [date(2026, 6, 8), date(2026, 6, 9)],
                    "fixing_value": [5.37, 5.39],
                }
            ),
            unique_by=("currency_pair", "ptax_fixing_date"),
        )
        .source(
            "curves",
            pl.DataFrame({"tenor_days": [21], "risk_free_rate": [0.12]}),
            unique_by=("tenor_days",),
        )
        .build()
    )
    enriched = bind(trades, market, UsdBrlDfInput)
    staged = graph.stage(enriched, view="output")
    priced = graph.plan(enriched, view="output")
    return priced, staged


@app.cell(hide_code=True)
def _(graph, mo, priced, staged):
    mo.vstack(
        [
            mo.md("## Grafo e fórmula\n```text\n" + graph.explain(view="output") + "\n```"),
            mo.md("## Debug: coluna de fixing usada"),
            mo.ui.table(
                staged.select("instrument_id", "tenor", "ptax_fixing_date", "ptax").collect()
            ),
            mo.md("## Preço final"),
            mo.ui.table(priced.collect()),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
