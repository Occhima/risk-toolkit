"""Posições em Forwards de Soja — MTM, Exposição e Delta.

Monta um book com três posições, precifica com price_forward(), computa o
delta por instrumento via grad() e levanta as medidas de posição com
position_value() e position_risk().

    uv run marimo edit docs/examples/02_soja_positions.py
    marimo export html docs/examples/02_soja_positions.py -o docs/examples/02_soja_positions.html
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Posições em Forwards de Soja — MTM, Exposição e Delta

        O Schenberg separa **pricing puro** de **posição**:

        - `price_forward()` devolve o valor por unidade do instrumento.
        - `grad(pv_expr, "forward_rate")` calcula ∂PV/∂F via JAX sobre os dados de mercado.
        - `position_value()` levanta: `exposure = side × qty`, `mtm = exposure × value`.
        - `position_risk()` levanta: `position_delta = exposure × Δ`.
        """
    )
    return


@app.cell
def _():
    from datetime import date

    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    from schenberg import MarketSnapshot, MarketSource, exp, grad, var
    from schenberg.domain.schemas.position import (
        BookContract,
        InstrumentRisk,
        InstrumentValue,
        Position,
        ReportingFx,
    )
    from schenberg.position import book_value_rollup, position_risk, position_value
    from schenberg.pricing.api import price_forward

    pl.Config.set_tbl_cols(-1)
    return (
        BookContract,
        InstrumentRisk,
        InstrumentValue,
        MarketSnapshot,
        MarketSource,
        Position,
        ReportingFx,
        date,
        exp,
        grad,
        jnp,
        mo,
        np,
        pl,
        plt,
        book_value_rollup,
        position_risk,
        position_value,
        price_forward,
        var,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Parâmetros de Mercado")
    return


@app.cell
def _(mo):
    fwd_slider = mo.ui.slider(800, 1400, value=1100, step=10, label="Preço Forward Soja (cents/bu)")
    rate_slider = mo.ui.slider(0.05, 0.20, value=0.105, step=0.005, label="Taxa r (a.a.)")
    usd_brl_slider = mo.ui.slider(4.5, 6.5, value=5.25, step=0.05, label="USD/BRL")
    mo.vstack([fwd_slider, rate_slider, usd_brl_slider])
    return fwd_slider, rate_slider, usd_brl_slider


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Book de Posições")
    return


@app.cell
def _(BookContract, Position):
    positions = Position.from_records(
        [
            {
                "position_id": "P-LONG-A",
                "book": "SOJA-BOOK",
                "instrument_type": "FORWARD",
                "instrument_id": "SOY-NOV27-K1000",
                "quantity": 500_000.0,
                "side": 1.0,
                "unit_notional": None,
            },
            {
                "position_id": "P-SHORT",
                "book": "SOJA-BOOK",
                "instrument_type": "FORWARD",
                "instrument_id": "SOY-NOV27-K1000",
                "quantity": 300_000.0,
                "side": -1.0,
                "unit_notional": None,
            },
            {
                "position_id": "P-LONG-B",
                "book": "SOJA-BOOK",
                "instrument_type": "FORWARD",
                "instrument_id": "SOY-NOV27-K1100",
                "quantity": 200_000.0,
                "side": 1.0,
                "unit_notional": None,
            },
        ]
    )

    book = BookContract.from_records(
        [
            {
                "book": "SOJA-BOOK",
                "desk": "Commodities",
                "legal_entity": "LE-BR",
                "reporting_currency": "BRL",
            }
        ]
    )
    return book, positions


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Pricing + Delta via `grad()`")
    return


@app.cell
def _(
    MarketSnapshot,
    MarketSource,
    date,
    exp,
    fwd_slider,
    grad,
    jnp,
    np,
    pl,
    price_forward,
    rate_slider,
    var,
):
    DAYS = 360
    F0 = float(fwd_slider.value)
    r0 = float(rate_slider.value)

    # Dois instrumentos com strikes distintos — mesma curva SOY
    trades = pl.DataFrame(
        {
            "instrument_id": ["SOY-NOV27-K1000", "SOY-NOV27-K1100"],
            "indexer": ["SOY", "SOY"],
            "currency": ["USD", "USD"],
            "strike": [1000.0, 1100.0],
            "payment_days": [DAYS, DAYS],
        }
    ).lazy()

    market = MarketSnapshot.from_sources(
        as_of=date(2026, 6, 6),
        sources=[
            MarketSource(
                "curves",
                pl.DataFrame(
                    {
                        "id_indexador": ["SOY"],
                        "tenor_days": [DAYS],
                        "forward_rate": [F0],
                        "risk_free_rate": [r0],
                    }
                ).lazy(),
            )
        ],
    )

    priced = price_forward(trades, market).collect()

    # Delta ∂PV/∂F via JAX — mesma expressão que o forward_formula usa
    pv_expr = (var("forward_rate") - var("strike")) * exp(
        -(var("risk_free_rate") * (var("payment_days") / 252.0))
    )
    delta_fn = grad(pv_expr, "forward_rate")

    deltas = np.array(
        delta_fn(
            {
                "forward_rate": jnp.array(priced["forward_rate"].to_numpy()),
                "strike": jnp.array(priced["strike"].to_numpy()),
                "risk_free_rate": jnp.array(priced["risk_free_rate"].to_numpy()),
                "payment_days": jnp.array(priced["payment_days"].cast(pl.Float64).to_numpy()),
            }
        )
    )

    priced, deltas
    return DAYS, F0, delta_fn, deltas, market, pv_expr, priced, r0, trades


@app.cell
def _(InstrumentRisk, InstrumentValue, deltas, priced):
    instrument_values = InstrumentValue.from_records(
        [
            {
                "instrument_type": "FORWARD",
                "instrument_id": row["instrument_id"],
                "value": row["value"],
                "currency": "USD",
            }
            for row in priced.to_dicts()
        ]
    )

    instrument_risk = InstrumentRisk.from_records(
        [
            {
                "instrument_type": "FORWARD",
                "instrument_id": row["instrument_id"],
                "currency": "USD",
                "delta": float(deltas[i]),
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho": 0.0,
            }
            for i, row in enumerate(priced.to_dicts())
        ]
    )
    return instrument_risk, instrument_values


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Position Value — MTM e Exposição")
    return


@app.cell
def _(
    ReportingFx,
    book,
    instrument_values,
    position_value,
    positions,
    usd_brl_slider,
):
    reporting_fx = ReportingFx.from_records(
        [
            {
                "currency": "USD",
                "reporting_currency": "BRL",
                "book_fx": 1.0 / float(usd_brl_slider.value),
            }
        ]
    )
    pos_val = position_value(
        positions, value=instrument_values, book=book, fx=reporting_fx
    ).collect()
    pos_val
    return pos_val, reporting_fx


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Position Risk — Delta por Posição")
    return


@app.cell
def _(instrument_risk, position_risk, positions):
    pos_risk = position_risk(positions, risk=instrument_risk).collect()
    pos_risk
    return (pos_risk,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Gráficos — MTM, Exposição e Delta")
    return


@app.cell
def _(F0, plt, pos_risk, pos_val, usd_brl_slider):
    consolidated = pos_val.join(
        pos_risk.select("position_id", "position_delta"), on="position_id", how="left"
    )

    ids = consolidated["position_id"].to_list()
    mtm_brl = consolidated["reported_mtm"].to_list()
    exposure = consolidated["exposure"].to_list()
    pos_delta = consolidated["position_delta"].to_list()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    def _bar(ax, vals, title, ylabel):
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
        ax.bar(ids, vals, color=colors, edgecolor="black", linewidth=0.7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, alpha=0.3, axis="y")

    _bar(axes[0], mtm_brl, "MTM (BRL)", "BRL")
    _bar(axes[1], exposure, "Exposição (bushels)", "side × qty")
    _bar(axes[2], pos_delta, "Delta de Posição", "exposure × ∂PV/∂F")

    fig.suptitle(
        f"SOJA-BOOK  |  F = {F0:.0f} cts/bu  |  USD/BRL = {usd_brl_slider.value:.2f}  "
        f"|  MTM Total = {sum(mtm_brl):,.0f} BRL  |  Delta Líquido = {sum(pos_delta):,.0f} bu",
        fontsize=10,
        fontweight="bold",
    )
    plt.tight_layout()
    fig
    return axes, consolidated, exposure, fig, ids, mtm_brl, pos_delta


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Book Rollup")
    return


@app.cell
def _(book_value_rollup, mo, pos_risk, pos_val):
    rollup = book_value_rollup.compute(pos_val.lazy()).collect()
    net_delta = sum(pos_risk["position_delta"].to_list())
    mo.md(
        f"""
        ### SOJA-BOOK

        | Métrica | Valor |
        |---------|-------|
        | **Exposição líquida** | {rollup["exposure"][0]:,.0f} bushels |
        | **MTM (BRL)** | {rollup["reported_mtm"][0]:,.0f} BRL |
        | **Delta líquido** | {net_delta:,.0f} bushels |
        """
    )
    return net_delta, rollup


if __name__ == "__main__":
    app.run()
