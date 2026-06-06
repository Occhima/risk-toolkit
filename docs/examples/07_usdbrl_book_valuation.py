"""Book valuation of a USD/BRL NDF — MTM, PnL and PnL explain, built on Schenberg.

A marimo notebook. One book holds the *same* USD/BRL non-deliverable forward both
**bought** (a long position) and **sold** (a short, smaller position). We price the
NDF with the Schenberg forward pricer, value the book with the position layer, then
explain the period PnL as a roll / curve / fx waterfall — repricing under
cumulative market moves, all with base Schenberg code. The market data is
fabricated for the demo.

    uv run marimo edit docs/examples/07_usdbrl_book_valuation.py
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # USD/BRL NDF book — value, PnL, PnL explain

        Pricing returns a **pure** per-instrument value; the **position layer** lifts it
        onto a book (`exposure = side × quantity`, reporting currency via FX). The same
        machinery lifts a per-instrument **PnL decomposition** into a position PnL explain.
        """
    )
    return


@app.cell
def _():
    from datetime import date

    import marimo as mo
    import polars as pl

    from schenberg.domain.schemas.position import (
        BookContract,
        InstrumentPnlExplain,
        Position,
        ReportingFx,
    )
    from schenberg.position import book_value_rollup, position_pnl_explain, position_value
    from schenberg.pricing.api import price_forward

    pl.Config.set_tbl_cols(-1)
    return (
        BookContract,
        InstrumentPnlExplain,
        Position,
        ReportingFx,
        book_value_rollup,
        date,
        mo,
        pl,
        position_pnl_explain,
        position_value,
        price_forward,
    )


@app.cell
def _(date, pl, price_forward):
    # ---- The NDF, and a Schenberg-native pricer that emits InstrumentValue ----
    INDEXER, STRIKE, INSTRUMENT_ID = "USDBRL", 5.00, "USDBRL-NDF"

    def ndf_market(*, fx_forward: float, rate: float, days: int):
        """Fabricated market: the USD/BRL forward, a discount rate, settled in BRL."""
        from schenberg.market_data.snapshot import MarketSnapshot
        from schenberg.market_data.sources import MarketSource

        return MarketSnapshot.from_sources(
            as_of=date(2026, 6, 5),
            sources=[
                MarketSource(
                    "curves",
                    pl.DataFrame(
                        {
                            "id_indexador": [INDEXER],
                            "tenor_days": [days],
                            "forward_rate": [fx_forward],
                            "risk_free_rate": [rate],
                        }
                    ).lazy(),
                ),
                # Settled in BRL: the forward formula's currency step is the identity.
                MarketSource(
                    "fx_rates",
                    pl.DataFrame({"currency": ["BRL"], "fx_rate": [1.0]}).lazy(),
                ),
            ],
        )

    def ndf_trades(*, days: int):
        return pl.DataFrame(
            {
                "instrument_id": [INSTRUMENT_ID],
                "tenor": [date(2027, 6, 5)],
                "indexer": [INDEXER],
                "currency": ["BRL"],
                "strike": [STRIKE],
                "payment_days": [days],
            }
        ).lazy()

    def price_ndf(*, fx_forward: float, rate: float, days: int):
        """Value the NDF (BRL) and shape it as an InstrumentValue, ready for the book."""
        market = ndf_market(fx_forward=fx_forward, rate=rate, days=days)
        priced = price_forward(ndf_trades(days=days), market)
        return priced.select(
            instrument_type=pl.lit("NDF"),
            instrument_id=pl.col("instrument_id"),
            value=pl.col("value"),
            currency=pl.lit("BRL"),
        )

    return INSTRUMENT_ID, ndf_trades, price_ndf


@app.cell
def _(BookContract, Position, ReportingFx, INSTRUMENT_ID):
    # ---- The book: the same NDF held long (bought) and short (sold) ----
    positions = Position.from_records(
        [
            {
                "position_id": "P-LONG",
                "book": "FX-BOOK",
                "instrument_type": "NDF",
                "instrument_id": INSTRUMENT_ID,
                "quantity": 1_000_000.0,  # USD notional bought
                "side": 1.0,
                "unit_notional": None,
            },
            {
                "position_id": "P-SHORT",
                "book": "FX-BOOK",
                "instrument_type": "NDF",
                "instrument_id": INSTRUMENT_ID,
                "quantity": 600_000.0,  # USD notional sold
                "side": -1.0,
                "unit_notional": None,
            },
        ]
    )
    book = BookContract.from_records(
        [{"book": "FX-BOOK", "desk": "EM FX", "legal_entity": "LE-BR", "reporting_currency": "BRL"}]
    )
    reporting_fx = ReportingFx.from_records(
        [{"currency": "BRL", "reporting_currency": "BRL", "book_fx": 1.0}]
    )
    return book, positions, reporting_fx


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 1 · Value the book

        `position_value` joins the spine (`Position`), the pure `InstrumentValue`, the
        `BookContract` and the `ReportingFx`, then exposes `exposure`, `mtm` and
        `reported_mtm` as measures. The short leg flips the sign of its MTM.
        """
    )
    return


@app.cell
def _(book, positions, price_ndf, reporting_fx, position_value):
    # t0 -> t1 market move: the FX forward and rate move, and the trade rolls 63 days.
    value_t0 = price_ndf(fx_forward=5.10, rate=0.100, days=252)
    value_t1 = price_ndf(fx_forward=5.25, rate=0.105, days=189)

    valued_t1 = position_value(positions, value=value_t1, book=book, fx=reporting_fx).collect()
    valued_t1
    return value_t0, value_t1, valued_t1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 2 · Period PnL

        The simplest PnL is the change in reported MTM between the two dates — one row per
        position, summed to the book.
        """
    )
    return


@app.cell
def _(book, positions, position_value, reporting_fx, value_t0, value_t1, pl):
    mtm_t0 = position_value(positions, value=value_t0, book=book, fx=reporting_fx)
    mtm_t1 = position_value(positions, value=value_t1, book=book, fx=reporting_fx)
    pnl = (
        mtm_t1.select("position_id", "book", reported_mtm_t1=pl.col("reported_mtm"))
        .join(
            mtm_t0.select("position_id", reported_mtm_t0=pl.col("reported_mtm")),
            on="position_id",
            how="inner",
        )
        .with_columns(pnl=pl.col("reported_mtm_t1") - pl.col("reported_mtm_t0"))
        .collect()
    )
    pnl
    return (pnl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 3 · PnL explain — a roll / curve / fx waterfall

        We attribute the value change by **repricing under cumulative market moves**: roll
        the trade forward in time, then move the discount curve, then move the FX forward.
        Each step's value delta is one explain component — additive by construction, so
        `total = roll + curve + fx`.
        """
    )
    return


@app.cell
def _(price_ndf, pl):
    def _value(**kw):
        return price_ndf(**kw).collect().select("value").item()

    # Cumulative states from t0 to t1.
    v0 = _value(fx_forward=5.10, rate=0.100, days=252)  # base (t0)
    v_roll = _value(fx_forward=5.10, rate=0.100, days=189)  # + time roll
    v_curve = _value(fx_forward=5.10, rate=0.105, days=189)  # + discount curve
    v_fx = _value(fx_forward=5.25, rate=0.105, days=189)  # + FX forward (== t1)

    roll_pnl = v_roll - v0
    curve_pnl = v_curve - v_roll
    fxfwd_pnl = v_fx - v_curve
    total_pnl = v_fx - v0
    waterfall = pl.DataFrame(
        {
            "component": ["roll", "curve", "fx", "total"],
            "value_pnl_per_unit": [roll_pnl, curve_pnl, fxfwd_pnl, total_pnl],
        }
    )
    waterfall
    return curve_pnl, fxfwd_pnl, roll_pnl, total_pnl


@app.cell
def _(
    InstrumentPnlExplain,
    INSTRUMENT_ID,
    book,
    curve_pnl,
    fxfwd_pnl,
    positions,
    position_pnl_explain,
    reporting_fx,
    roll_pnl,
    total_pnl,
):
    # Assemble the pure per-instrument decomposition, then lift it onto the book:
    # each position component is exposure * <component> / book_fx.
    instrument_pnl = InstrumentPnlExplain.from_records(
        [
            {
                "instrument_type": "NDF",
                "instrument_id": INSTRUMENT_ID,
                "currency": "BRL",
                "roll_value_pnl": roll_pnl,
                "curve_value_pnl": curve_pnl,
                "fx_value_pnl": fxfwd_pnl,
                "fixing_value_pnl": 0.0,
                "residual_value_pnl": total_pnl - (roll_pnl + curve_pnl + fxfwd_pnl),
                "total_value_pnl": total_pnl,
            }
        ]
    )
    book_pnl_explain = position_pnl_explain(
        positions, pnl=instrument_pnl, book=book, fx=reporting_fx
    ).collect()
    book_pnl_explain
    return (book_pnl_explain,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 4 · Roll up to the book

        Book-level aggregation is a *later* layer — a `Fold` over the per-position rows,
        never part of the view. The long and short legs net into one book PnL.
        """
    )
    return


@app.cell
def _(book_value_rollup, book, positions, position_value, reporting_fx, value_t1):
    book_value = book_value_rollup.compute(
        position_value(positions, value=value_t1, book=book, fx=reporting_fx)
    ).collect()
    book_value
    return


if __name__ == "__main__":
    app.run()
