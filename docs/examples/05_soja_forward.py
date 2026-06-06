"""Precificação de forward de soja CBOT — vencimento novembro 2027.

Notebook Marimo. Mostra o MTM e o delta de um forward de soja em função do
preço do ativo, usando a API do Schenberg.

    uv run marimo edit docs/examples/05_soja_forward.py
    marimo export html docs/examples/05_soja_forward.py -o docs/examples/05_soja_forward.html
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Forward de Soja CBOT — Vencimento Novembro 2027

        Um **forward de soja** é um contrato que obriga o comprador a adquirir
        um volume de soja a um preço fixo (strike K) em uma data futura.

        O valor presente do contrato é:

        $$
        PV = (F - K) \cdot e^{-rT}
        $$

        onde **F** é o preço forward da soja hoje, **K** é o strike, **r** é a
        taxa livre de risco e **T** é o prazo em anos.

        O **delta** mede a sensibilidade do PV a variações no preço forward:

        $$
        \Delta = \frac{\partial PV}{\partial F} = e^{-rT}
        $$

        Para um forward, o delta é **constante** (independe do nível de preço), o
        que o distingue de opções. Abaixo exploramos o perfil de MTM, delta e a
        sensibilidade temporal usando preços realistas do mercado CBOT.
        """
    )
    return


@app.cell
def _():
    from datetime import date, timedelta

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    from schenberg.market_data.snapshot import MarketSnapshot
    from schenberg.market_data.sources import MarketSource
    from schenberg.pricing.api import price_forward

    pl.Config.set_tbl_cols(-1)
    return MarketSnapshot, MarketSource, date, mo, np, pl, plt, price_forward, timedelta


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Parâmetros do Contrato

        Ajuste os sliders para explorar o impacto de cada variável no pricing.
        """
    )
    return


@app.cell
def _(mo):
    strike_slider = mo.ui.slider(800, 1400, value=1050, step=10, label="Strike K (cents/bushel)")
    rate_slider = mo.ui.slider(0.05, 0.20, value=0.105, step=0.005, label="Taxa CDI r (a.a.)")
    days_slider = mo.ui.slider(63, 500, value=360, step=21, label="Dias úteis até vencimento")
    mo.vstack([strike_slider, rate_slider, days_slider])
    return days_slider, rate_slider, strike_slider


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Dados de Mercado")
    return


@app.cell
def _(MarketSnapshot, MarketSource, date, days_slider, pl, rate_slider):
    # Preço forward CBOT Soja Novembro 2027 (SRY27)
    # Referência: aproximadamente 1100 cents/bushel em jun/2026
    INDEXER = "SOY"
    FORWARD_PRICE = 1100.0  # cents/bushel — preço de mercado CBOT NOV27

    market = MarketSnapshot.from_sources(
        as_of=date(2026, 6, 6),
        sources=[
            MarketSource(
                "curves",
                pl.DataFrame(
                    {
                        "id_indexador": [INDEXER],
                        "tenor_days": [days_slider.value],
                        "forward_rate": [FORWARD_PRICE],
                        "risk_free_rate": [rate_slider.value],
                    }
                ).lazy(),
            ),
        ],
    )
    market
    return FORWARD_PRICE, INDEXER, market


@app.cell
def _(FORWARD_PRICE, INDEXER, days_slider, pl, price_forward, market, strike_slider):
    # Precifica um único contrato forward de soja
    trades = pl.DataFrame(
        {
            "instrument_id": ["SOY-NOV27"],
            "indexer": [INDEXER],
            "currency": ["USD"],
            "strike": [float(strike_slider.value)],
            "payment_days": [days_slider.value],
        }
    ).lazy()

    priced = price_forward(trades, market).collect()

    T = days_slider.value / 252.0
    import math

    delta_unit = math.exp(-market.source("curves").data.collect()["risk_free_rate"][0] * T)

    mo_summary = {
        "Preço Forward (F)": f"{FORWARD_PRICE:.0f} cts/bu",
        "Strike (K)": f"{strike_slider.value:.0f} cts/bu",
        "Dias úteis": days_slider.value,
        "Prazo (anos)": f"{T:.2f}",
        "FV = F - K": f"{priced['future_value'][0]:.2f} cts/bu",
        "PV (valor presente)": f"{priced['present_value'][0]:.2f} cts/bu",
        "Delta (∂PV/∂F)": f"{delta_unit:.4f}",
    }
    priced, mo_summary
    return T, delta_unit, mo_summary, priced, trades


@app.cell
def _(mo, mo_summary):
    mo.md(
        f"""
        ### Resultado do Pricing

        | Métrica | Valor |
        |---------|-------|
        | **Preço Forward (F)** | {mo_summary["Preço Forward (F)"]} |
        | **Strike (K)** | {mo_summary["Strike (K)"]} |
        | **Dias úteis** | {mo_summary["Dias úteis"]} |
        | **Prazo** | {mo_summary["Prazo (anos)"]} anos |
        | **FV = F − K** | {mo_summary["FV = F - K"]} |
        | **PV (valor presente)** | {mo_summary["PV (valor presente)"]} |
        | **Delta** | {mo_summary["Delta (∂PV/∂F)"]} |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Perfil de MTM — Valor presente pelo preço da soja

        O gráfico abaixo mostra o **valor presente (MTM)** do forward em função
        do preço forward da soja. A relação é linear — diferente de uma opção —
        e o slope é exatamente o delta = eˉʳᵀ.
        """
    )
    return


@app.cell
def _(
    INDEXER,
    MarketSnapshot,
    MarketSource,
    days_slider,
    date,
    np,
    pl,
    plt,
    price_forward,
    rate_slider,
    strike_slider,
):
    F_range = np.linspace(700, 1500, 200)
    K = float(strike_slider.value)
    r = float(rate_slider.value)
    T_val = days_slider.value / 252.0
    import math as _math

    DF = _math.exp(-r * T_val)
    pv_values = (F_range - K) * DF

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(F_range, pv_values, color="#1f77b4", linewidth=2.5, label="PV = (F − K)·e⁻ʳᵀ")
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.axvline(K, color="#d62728", linewidth=1.2, linestyle=":", label=f"Strike K = {K:.0f}")
    ax1.axvline(1100, color="#ff7f0e", linewidth=1.2, linestyle=":", label="Mercado F = 1100")
    ax1.fill_between(F_range, pv_values, 0, where=pv_values > 0, alpha=0.15, color="#2ca02c")
    ax1.fill_between(F_range, pv_values, 0, where=pv_values < 0, alpha=0.15, color="#d62728")
    ax1.set_xlabel("Preço Forward da Soja F (cents/bushel)", fontsize=12)
    ax1.set_ylabel("Valor Presente — PV (cents/bushel)", fontsize=12)
    ax1.set_title("Perfil de MTM — Forward de Soja NOV27", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1
    return DF, F_range, K, T_val, ax1, fig1, pv_values, r


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Delta pelo Preço da Soja

        Para um forward linear, o **delta é constante** (não depende de F).
        Isso contrasta com uma opção, cujo delta varia com a moneyness.

        O gráfico abaixo compara o delta do forward com o delta de uma call
        ATM hipotética (Black-Scholes) para o mesmo prazo, destacando a
        diferença estrutural.
        """
    )
    return


@app.cell
def _(DF, F_range, K, T_val, np, plt, r):
    # Delta do forward: constante = e^(-rT)
    delta_fwd = np.full_like(F_range, DF)

    # Delta de uma call (Black-Scholes) com vol = 20% para comparação visual
    sigma = 0.20
    d1 = (np.log(F_range / K) + 0.5 * sigma**2 * T_val) / (sigma * np.sqrt(T_val))
    import math as _m_norm

    delta_call = np.array([0.5 * _m_norm.erfc(-x / _m_norm.sqrt(2)) for x in d1])

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(
        F_range, delta_fwd, color="#1f77b4", linewidth=2.5, label=f"Forward Δ = e⁻ʳᵀ = {DF:.3f}"
    )
    ax2.plot(
        F_range,
        delta_call,
        color="#ff7f0e",
        linewidth=2,
        linestyle="--",
        label="Call ATM Δ (σ=20%)",
    )
    ax2.axvline(K, color="#d62728", linewidth=1.2, linestyle=":", label=f"Strike K = {K:.0f}")
    ax2.axvline(1100, color="gray", linewidth=1, linestyle=":", alpha=0.7, label="F = 1100")
    ax2.set_xlabel("Preço Forward da Soja F (cents/bushel)", fontsize=12)
    ax2.set_ylabel("Delta (∂PV/∂F)", fontsize=12)
    ax2.set_title("Delta pelo Preço da Soja — Forward vs. Call", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2
    return ax2, d1, delta_call, delta_fwd, fig2, sigma


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Delta vs. Prazo ao Vencimento

        À medida que o contrato se aproxima do vencimento, o delta do forward
        aumenta de e⁻ʳᵀ em direção a 1.0. Quanto mais curto o prazo, mais o
        forward se comporta como uma posição no ativo subjacente puro.
        """
    )
    return


@app.cell
def _(np, plt, r):
    days_grid = np.linspace(5, 504, 300)
    T_grid = days_grid / 252.0
    import math as _m2

    delta_grid = np.array([_m2.exp(-r * t) for t in T_grid])

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    ax3.plot(days_grid, delta_grid, color="#2ca02c", linewidth=2.5)
    ax3.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4, label="Δ = 1 (spot)")
    ax3.axvline(360, color="#d62728", linewidth=1.2, linestyle=":", label="Hoje (360 d.u.)")
    ax3.fill_between(days_grid, delta_grid, 1.0, alpha=0.1, color="#2ca02c")
    ax3.set_xlabel("Dias úteis até o vencimento", fontsize=12)
    ax3.set_ylabel("Delta (∂PV/∂F)", fontsize=12)
    ax3.set_title("Delta vs. Prazo ao Vencimento — Forward NOV27", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    fig3
    return T_grid, ax3, days_grid, delta_grid, fig3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Mapa de Sensibilidade — PV por (F, r)

        O heatmap abaixo mostra o valor presente do forward para diferentes
        combinações de preço forward e taxa de desconto. As isolinhas representam
        PV = 0 (breakeven).
        """
    )
    return


@app.cell
def _(K, T_val, np, plt):

    F_grid = np.linspace(800, 1400, 60)
    r_grid = np.linspace(0.05, 0.18, 50)
    FF, RR = np.meshgrid(F_grid, r_grid)
    PV_grid = (FF - K) * np.exp(-RR * T_val)

    fig4, ax4 = plt.subplots(figsize=(9, 6))
    c = ax4.contourf(FF, RR, PV_grid, levels=30, cmap="RdYlGn")
    cs = ax4.contour(FF, RR, PV_grid, levels=[0], colors="black", linewidths=2)
    ax4.clabel(cs, fmt="PV=0", fontsize=9)
    plt.colorbar(c, ax=ax4, label="Valor Presente (cents/bushel)")
    ax4.set_xlabel("Preço Forward da Soja F (cents/bushel)", fontsize=12)
    ax4.set_ylabel("Taxa de desconto r (a.a.)", fontsize=12)
    ax4.set_title(
        f"Mapa de PV — Forward Soja NOV27  (K={K:.0f} cts/bu)", fontsize=13, fontweight="bold"
    )
    ax4.plot(1100, 0.105, "k*", markersize=12, label="Ponto de mercado (F=1100, r=10.5%)")
    ax4.legend(fontsize=10)
    plt.tight_layout()
    fig4
    return FF, PV_grid, RR, ax4, c, cs, fig4, r_grid


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Fórmula do Grafo — Inspeção do Motor de Pricing

        O Schenberg mantém o grafo de fórmulas completamente inspecionável.
        Abaixo: os nós, a ordem topológica e o diagrama Mermaid.
        """
    )
    return


@app.cell
def _(mo):
    from schenberg.pricing.api import forward_formula

    inputs_str = str(sorted(forward_formula.required_inputs("output")))
    formulas_str = "\n".join(f"  {v}" for v in forward_formula.formulas().values())
    mermaid_str = forward_formula.to_mermaid()
    mo.vstack(
        [
            mo.md(f"**Inputs necessários:**\n\n```\n{inputs_str}\n```"),
            mo.md(f"**Fórmulas (LaTeX):**\n\n```\n{formulas_str}\n```"),
            mo.md(f"**Mermaid flowchart:**\n\n```\n{mermaid_str}\n```"),
        ]
    )
    return (forward_formula,)


if __name__ == "__main__":
    app.run()
