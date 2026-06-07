"""Posição sobre forwards precificados com FormulaGraph decorado.

Execute with `uv run poe examples-html`.
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Posição sobre forwards

        A direção econômica (`side`) fica somente na camada de posição. O grafo
        puro de preço gera `value` e `delta` por unidade, sem comprado/vendido.
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
    from schenberg.domain.schemas.position import (
        BookContract,
        InstrumentRisk,
        InstrumentValue,
        Position,
        ReportingFx,
    )
    from schenberg.position import book_value_rollup, position_risk, position_value

    return (
        BookContract,
        FormulaGraph,
        InstrumentRisk,
        InstrumentValue,
        MarketSnapshot,
        Position,
        ReportingFx,
        SchenbergDataFrameModel,
        With,
        bind,
        book_value_rollup,
        date,
        exp,
        market_role,
        mo,
        pl,
        position_risk,
        position_value,
    )


@app.cell
def _(FormulaGraph, SchenbergDataFrameModel, With, exp, market_role):
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

    graph = FormulaGraph("forward_position_example", input=ForwardInput)

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
        value="present_value",
        delta="delta",
        currency="currency",
    )
    return ForwardInput, graph


@app.cell
def _(ForwardInput, MarketSnapshot, bind, date, graph, pl):
    trades = pl.DataFrame(
        {
            "instrument_id": ["FWD-SOY-1", "FWD-SOY-2"],
            "indexer": ["SOY", "SOY"],
            "currency": ["USD", "USD"],
            "strike": [1000.0, 1040.0],
            "payment_days": [252, 504],
        }
    ).lazy()
    market = (
        MarketSnapshot.at(date(2026, 6, 6))
        .source(
            "curves",
            pl.DataFrame(
                {
                    "id_indexador": ["SOY", "SOY"],
                    "tenor_days": [252, 504],
                    "forward_rate": [1050.0, 1120.0],
                    "risk_free_rate": [0.10, 0.12],
                }
            ),
            unique_by=("id_indexador", "tenor_days"),
        )
        .build()
    )
    priced = graph.plan(bind(trades, market, ForwardInput), view="output").collect()
    return priced


@app.cell
def _(BookContract, InstrumentRisk, InstrumentValue, Position, ReportingFx, pl, priced):
    positions = Position.from_records(
        [
            {
                "position_id": "P1",
                "book": "COMMOD",
                "instrument_type": "FORWARD",
                "instrument_id": "FWD-SOY-1",
                "quantity": 100.0,
                "side": 1.0,
                "unit_notional": 1.0,
            },
            {
                "position_id": "P2",
                "book": "COMMOD",
                "instrument_type": "FORWARD",
                "instrument_id": "FWD-SOY-2",
                "quantity": 40.0,
                "side": -1.0,
                "unit_notional": 1.0,
            },
        ]
    )
    values = InstrumentValue.from_polars(
        priced.select(
            instrument_type=pl.lit("FORWARD"),
            instrument_id=pl.col("instrument_id"),
            value=pl.col("value"),
            currency=pl.col("currency"),
        )
    )
    risks = InstrumentRisk.from_polars(
        priced.select(
            instrument_type=pl.lit("FORWARD"),
            instrument_id=pl.col("instrument_id"),
            currency=pl.col("currency"),
            delta=pl.col("delta"),
            gamma=pl.lit(0.0),
            vega=pl.lit(0.0),
            theta=pl.lit(0.0),
            rho=pl.lit(0.0),
        )
    )
    book = BookContract.from_records(
        [
            {
                "book": "COMMOD",
                "desk": "Commodities",
                "legal_entity": "LE-BR",
                "reporting_currency": "BRL",
            }
        ]
    )
    fx = ReportingFx.from_records(
        [{"currency": "USD", "reporting_currency": "BRL", "book_fx": 0.20}]
    )
    return book, fx, positions, risks, values


@app.cell
def _(book, book_value_rollup, fx, position_risk, position_value, positions, risks, values):
    position_values = position_value(positions, value=values, book=book, fx=fx).collect()
    risk_values = position_risk(positions, risk=risks).collect()
    book_rollup = book_value_rollup.compute(position_values.lazy()).collect()
    return book_rollup, position_values, risk_values


@app.cell(hide_code=True)
def _(book_rollup, graph, mo, position_values, priced, risk_values):
    mo.vstack(
        [
            mo.md(
                "## Grafo puro de preço (sem `side`)\n```text\n"
                + graph.explain(view="output")
                + "\n```"
            ),
            mo.md("## Instrumentos precificados"),
            mo.ui.table(priced),
            mo.md("## Posição: MTM e exposição"),
            mo.ui.table(position_values),
            mo.md("## Posição: delta levantado por exposição"),
            mo.ui.table(risk_values),
            mo.md("## Agregação por book"),
            mo.ui.table(book_rollup),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
