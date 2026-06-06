"""View de posição — MTM, exposição e delta de um book de soja.

Notebook Marimo. Monta um book com posições compradas e vendidas em forwards
de soja, computa MTM e delta usando a API do Schenberg, e exibe os resultados
com gráficos de barra.

    uv run marimo edit docs/examples/06_position_view.py
    marimo export html docs/examples/06_position_view.py -o docs/examples/06_position_view.html
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # View de Posição — Book de Soja

        O Schenberg separa **pricing puro** de **posição**:

        - `price_forward` retorna o valor de *uma unidade* do instrumento — sem
          quantidade, sem direção, sem book.
        - `position_value` levanta esse valor sobre a posição: `exposure = side × qty`,
          `mtm = exposure × value`.
        - `position_risk` levanta o delta por instrumento sobre a posição:
          `position_delta = exposure × delta`.

        Aqui temos um book com três posições em forwards de soja CBOT NOV27
        (comprada, vendida e outra comprada). Calculamos MTM, exposição e delta
        para cada posição e para o book consolidado.
        """
    )
    return


@app.cell
def _():
    from datetime import date

    import marimo as mo
    import matplotlib.pyplot as plt
    import math
    import polars as pl

    from schenberg.domain.schemas.position import (
        BookContract,
        InstrumentRisk,
        InstrumentValue,
        Position,
        ReportingFx,
    )
    from schenberg.market_data.snapshot import MarketSnapshot
    from schenberg.market_data.sources import MarketSource
    from schenberg.position import position_risk, position_value
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
        math,
        mo,
        pl,
        plt,
        position_risk,
        position_value,
        price_forward,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 1 · Parâmetros de Mercado")
    return


@app.cell
def _(mo):
    fwd_slider = mo.ui.slider(800, 1400, value=1100, step=10, label="Preço Forward Soja (cents/bu)")
    rate_slider = mo.ui.slider(0.05, 0.20, value=0.105, step=0.005, label="Taxa CDI r (a.a.)")
    usd_brl_slider = mo.ui.slider(4.5, 6.5, value=5.25, step=0.05, label="USD/BRL")
    mo.vstack([fwd_slider, rate_slider, usd_brl_slider])
    return fwd_slider, rate_slider, usd_brl_slider


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 2 · Posições no Book")
    return


@app.cell
def _(Position, BookContract, ReportingFx):
    positions = Position.from_records(
        [
            {
                "position_id": "P-LONG-A",
                "book": "SOJA-BOOK",
                "instrument_type": "FORWARD",
                "instrument_id": "SOY-NOV27",
                "quantity": 500_000.0,  # 500 mil bushels comprados
                "side": 1.0,
                "unit_notional": None,
            },
            {
                "position_id": "P-SHORT",
                "book": "SOJA-BOOK",
                "instrument_type": "FORWARD",
                "instrument_id": "SOY-NOV27",
                "quantity": 300_000.0,  # 300 mil bushels vendidos
                "side": -1.0,
                "unit_notional": None,
            },
            {
                "position_id": "P-LONG-B",
                "book": "SOJA-BOOK",
                "instrument_type": "FORWARD",
                "instrument_id": "SOY-NOV27-K1100",
                "quantity": 200_000.0,  # 200 mil bushels comprados (strike diferente)
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

    # Conversão USD → BRL (será atualizada pelo slider)
    return book, positions


@app.cell
def _(
    MarketSnapshot,
    MarketSource,
    date,
    fwd_slider,
    pl,
    price_forward,
    rate_slider,
):
    # Dois instrumentos com strikes diferentes
    DAYS = 360
    market_soja = MarketSnapshot.from_sources(
        as_of=date(2026, 6, 6),
        sources=[
            MarketSource(
                "curves",
                pl.DataFrame(
                    {
                        "id_indexador": ["SOY", "SOY"],
                        "tenor_days": [DAYS, DAYS],
                        # Dois registros para dois instrumentos — na prática seriam
                        # curvas completas; aqui simplificamos com ponto único.
                        "forward_rate": [float(fwd_slider.value), float(fwd_slider.value)],
                        "risk_free_rate": [float(rate_slider.value), float(rate_slider.value)],
                    }
                ).lazy(),
            ),
        ],
    )

    # Dois contratos: strike 1000 e strike 1100
    soja_trades = pl.DataFrame(
        {
            "instrument_id": ["SOY-NOV27", "SOY-NOV27-K1100"],
            "indexer": ["SOY", "SOY"],
            "currency": ["USD", "USD"],
            "strike": [1000.0, 1100.0],
            "payment_days": [DAYS, DAYS],
        }
    ).lazy()

    priced_soja = price_forward(soja_trades, market_soja).collect()
    priced_soja
    return DAYS, market_soja, priced_soja, soja_trades


@app.cell
def _(InstrumentValue, math, pl, priced_soja, rate_slider, DAYS):
    import math as _math

    T_pos = DAYS / 252.0
    delta_per_unit = _math.exp(-float(rate_slider.value) * T_pos)

    instrument_values = InstrumentValue.from_records(
        [
            {
                "instrument_type": "FORWARD",
                "instrument_id": row["instrument_id"],
                "value": row["value"],
                "currency": "USD",
            }
            for row in priced_soja.to_dicts()
        ]
    )
    instrument_values
    return T_pos, delta_per_unit, instrument_values


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 3 · Position Value — MTM e Exposição")
    return


@app.cell
def _(
    BookContract,
    ReportingFx,
    book,
    instrument_values,
    pl,
    position_value,
    positions,
    usd_brl_slider,
):
    reporting_fx = ReportingFx.from_records(
        [
            {
                "currency": "USD",
                "reporting_currency": "BRL",
                # book_fx: 1 USD = N BRL → value_brl = value_usd / (1/N) = value_usd * N
                # Na convenção do Schenberg: reported_mtm = mtm / book_fx
                # Logo book_fx = 1/usd_brl → reported_mtm = mtm * usd_brl
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
    mo.md("## 4 · Position Risk — Delta por Posição")
    return


@app.cell
def _(InstrumentRisk, delta_per_unit, pl, position_risk, positions, priced_soja):
    instrument_risk = InstrumentRisk.from_records(
        [
            {
                "instrument_type": "FORWARD",
                "instrument_id": row["instrument_id"],
                "currency": "USD",
                "delta": delta_per_unit,  # e^(-rT) — igual para ambos os contratos
                "gamma": 0.0,
                "vega": 0.0,
                "theta": -row["value"] * 0.105 / 252.0,  # theta ≈ -r·PV/252
                "rho": -row["value"] * (360 / 252.0),  # rho ≈ -T·PV
            }
            for row in priced_soja.to_dicts()
        ]
    )

    pos_risk = position_risk(positions, risk=instrument_risk).collect()
    pos_risk
    return instrument_risk, pos_risk


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 5 · Consolidado — MTM, Exposição e Delta por Posição")
    return


@app.cell
def _(pl, pos_risk, pos_val):
    consolidated = pos_val.join(
        pos_risk.select("position_id", "position_delta"),
        on="position_id",
        how="left",
    )
    consolidated
    return (consolidated,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 6 · Gráficos — MTM, Exposição e Delta

        Os gráficos abaixo mostram as três métricas por posição e o total do book.
        Posições vendidas (side = −1) aparecem com valores negativos, refletindo
        a direção econômica correta.
        """
    )
    return


@app.cell
def _(consolidated, plt, usd_brl_slider):
    ids = consolidated["position_id"].to_list()
    mtm_brl = consolidated["reported_mtm"].to_list()
    exposure = consolidated["exposure"].to_list()
    pos_delta = consolidated["position_delta"].to_list()

    fig_bars, axes = plt.subplots(1, 3, figsize=(14, 5))

    colors_mtm = ["#2ca02c" if v >= 0 else "#d62728" for v in mtm_brl]
    colors_exp = ["#1f77b4" if v >= 0 else "#ff7f0e" for v in exposure]
    colors_del = ["#9467bd" if v >= 0 else "#8c564b" for v in pos_delta]

    axes[0].bar(ids, mtm_brl, color=colors_mtm, edgecolor="black", linewidth=0.7)
    axes[0].set_title("MTM (BRL)", fontweight="bold")
    axes[0].set_ylabel("BRL (cents × bushels × USD/BRL)")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(ids, exposure, color=colors_exp, edgecolor="black", linewidth=0.7)
    axes[1].set_title("Exposição (bushels)", fontweight="bold")
    axes[1].set_ylabel("side × quantity (bushels)")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(ids, pos_delta, color=colors_del, edgecolor="black", linewidth=0.7)
    axes[2].set_title("Delta de Posição", fontweight="bold")
    axes[2].set_ylabel("exposure × delta_unit")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].tick_params(axis="x", rotation=20)

    usd_brl_val = float(usd_brl_slider.value)
    total_mtm = sum(mtm_brl)
    net_exposure = sum(exposure)
    net_delta = sum(pos_delta)

    fig_bars.suptitle(
        f"Book Consolidado — Soja NOV27  |  USD/BRL = {usd_brl_val:.2f}  "
        f"|  MTM Total = {total_mtm:,.0f} BRL  "
        f"|  Delta Líquido = {net_delta:,.0f} bu",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    fig_bars
    return (
        axes,
        colors_del,
        colors_exp,
        colors_mtm,
        exposure,
        fig_bars,
        ids,
        mtm_brl,
        net_delta,
        net_exposure,
        pos_delta,
        total_mtm,
        usd_brl_val,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 7 · Rollup do Book")
    return


@app.cell
def _(mo, net_delta, net_exposure, pos_val, total_mtm, usd_brl_val):
    from schenberg.position import book_value_rollup

    rollup = book_value_rollup.compute(pos_val.lazy()).collect()

    mo.md(
        f"""
        ### Book: SOJA-BOOK

        | Métrica | Valor |
        |---------|-------|
        | **Exposição líquida** | {net_exposure:,.0f} bushels |
        | **MTM (BRL)** | {total_mtm:,.0f} BRL |
        | **Delta líquido** | {net_delta:,.0f} bushels |
        | **USD/BRL** | {usd_brl_val:.2f} |

        A exposição líquida reflete o neto comprado/vendido. O delta líquido mostra
        quantos bushels de soja spot seria necessário para neutralizar a posição
        (hedge de delta).
        """
    )
    return (book_value_rollup, rollup)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 8 · Inspeção da View

        As views do Schenberg são declarativas e autoexplicativas.
        """
    )
    return


@app.cell
def _(mo, position_risk, position_value):
    pv_explain = position_value.explain()
    pr_explain = position_risk.explain()
    mo.vstack(
        [
            mo.md(f"**position_value.explain():**\n\n```\n{pv_explain}\n```"),
            mo.md(f"**position_risk.explain():**\n\n```\n{pr_explain}\n```"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
