"""Pricer de Forward de Soja — FormulaGraph + delta via JAX autodiff.

Cria o MarketSnapshot, precifica com price_forward() e computa o delta
∂PV/∂F usando grad() da Expr IR compilada via JAX.

    uv run marimo edit docs/examples/01_soja_pricer.py
    marimo export html docs/examples/01_soja_pricer.py -o docs/examples/01_soja_pricer.html
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Pricer de Forward de Soja — Grafo + Delta via JAX

        O Schenberg representa a fórmula como uma árvore simbólica (`Expr`). A mesma
        árvore é compilada para **Polars** (execução vetorizada) e para **JAX autodiff**
        — `grad(expr, "forward_rate")` retorna ∂PV/∂F sem derivar à mão.

        Para um forward linear:

        $$PV = (F - K) \cdot e^{-rT}$$

        $$\Delta = \frac{\partial PV}{\partial F} = e^{-rT}$$
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

    from schenberg import MarketSnapshot, MarketSource, bind, exp, grad, var
    from schenberg.pricing.api import ForwardContractPricing, forward_formula, price_forward

    pl.Config.set_tbl_cols(-1)
    return (
        ForwardContractPricing,
        MarketSnapshot,
        MarketSource,
        bind,
        date,
        exp,
        forward_formula,
        grad,
        jnp,
        mo,
        np,
        pl,
        plt,
        price_forward,
        var,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Parâmetros do Contrato")
    return


@app.cell
def _(mo):
    strike_slider = mo.ui.slider(800, 1400, value=1050, step=10, label="Strike K (cents/bushel)")
    rate_slider = mo.ui.slider(0.05, 0.20, value=0.105, step=0.005, label="Taxa r (a.a.)")
    days_slider = mo.ui.slider(63, 504, value=360, step=21, label="Dias úteis até vencimento")
    mo.vstack([strike_slider, rate_slider, days_slider])
    return days_slider, rate_slider, strike_slider


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Grafo de Fórmulas

        O `forward_formula` é o `FormulaGraph` que o `price_forward()` usa internamente.
        Inspecionável: fórmulas em LaTeX derivadas do próprio grafo, e diagrama de dependências.
        """
    )
    return


@app.cell
def _(forward_formula, mo):
    formulas_latex = "\n".join(f"  {v}" for v in forward_formula.formulas().values())
    mermaid_src = forward_formula.to_mermaid()
    mo.vstack(
        [
            mo.md(f"**Fórmulas (LaTeX gerado pelo grafo):**\n\n```\n{formulas_latex}\n```"),
            mo.md(f"**Dependências:**\n\n```\n{mermaid_src}\n```"),
        ]
    )
    return (formulas_latex, mermaid_src)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Expr IR + `grad()` — Delta analítico via JAX

        Constrói a expressão PV com `var()` / `exp()` e diferencia com `grad()`.
        A mesma estrutura de árvore que o `forward_formula` usa internamente.
        """
    )
    return


@app.cell
def _(days_slider, exp, grad, jnp, np, rate_slider, strike_slider, var):
    # Expressão simbólica — mesma lógica do _build_forward_graph em api.py
    pv_expr = (var("forward_rate") - var("strike")) * exp(
        -(var("risk_free_rate") * (var("payment_days") / 252.0))
    )

    # Derivada ∂PV/∂F compilada via JAX
    delta_fn = grad(pv_expr, "forward_rate")

    K_val = float(strike_slider.value)
    r_val = float(rate_slider.value)
    T_days = int(days_slider.value)

    # Avalia o delta em uma faixa de preços do ativo
    N = 200
    F_scan = jnp.linspace(700.0, 1500.0, N)
    delta_values = np.array(
        delta_fn(
            {
                "forward_rate": F_scan,
                "strike": jnp.full(N, K_val),
                "risk_free_rate": jnp.full(N, r_val),
                "payment_days": jnp.full(N, float(T_days)),
            }
        )
    )
    return F_scan, K_val, N, T_days, delta_fn, delta_values, pv_expr, r_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## MTM e Delta vs. Preço do Ativo

        **PV**: gerado com `price_forward()` sobre 200 instrumentos sintéticos —
        um MarketSnapshot com 200 entradas de curva, uma por ponto da faixa de F.

        **Delta**: saída de `grad(pv_expr, "forward_rate")` com JAX — constante
        em e⁻ʳᵀ para o forward linear.
        """
    )
    return


@app.cell
def _(
    F_scan,
    K_val,
    MarketSnapshot,
    MarketSource,
    N,
    T_days,
    date,
    delta_values,
    np,
    pl,
    plt,
    price_forward,
    r_val,
):
    scan_ids = [f"S{i:03d}" for i in range(N)]

    trades_scan = pl.DataFrame(
        {
            "instrument_id": [f"SOY-SCAN-{s}" for s in scan_ids],
            "indexer": scan_ids,
            "currency": ["USD"] * N,
            "strike": [K_val] * N,
            "payment_days": [T_days] * N,
        }
    ).lazy()

    market_scan = MarketSnapshot.from_sources(
        as_of=date(2026, 6, 6),
        sources=[
            MarketSource(
                "curves",
                pl.DataFrame(
                    {
                        "id_indexador": scan_ids,
                        "tenor_days": [T_days] * N,
                        "forward_rate": np.array(F_scan).tolist(),
                        "risk_free_rate": [r_val] * N,
                    }
                ).lazy(),
            )
        ],
    )

    priced_scan = price_forward(trades_scan, market_scan).collect()
    pv_values = priced_scan["value"].to_numpy()
    F_np = np.array(F_scan)

    fig, (ax_pv, ax_delta) = plt.subplots(1, 2, figsize=(13, 5))

    # MTM profile
    ax_pv.plot(F_np, pv_values, color="#1f77b4", linewidth=2.5)
    ax_pv.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax_pv.axvline(K_val, color="#d62728", linewidth=1.2, linestyle=":", label=f"K = {K_val:.0f}")
    ax_pv.fill_between(F_np, pv_values, 0, where=pv_values > 0, alpha=0.12, color="#2ca02c")
    ax_pv.fill_between(F_np, pv_values, 0, where=pv_values < 0, alpha=0.12, color="#d62728")
    ax_pv.set_xlabel("Preço Forward F (cents/bushel)")
    ax_pv.set_ylabel("PV (cents/bushel)")
    ax_pv.set_title("MTM — price_forward() sobre 200 instrumentos")
    ax_pv.legend()
    ax_pv.grid(True, alpha=0.3)

    # Delta profile
    ax_delta.plot(
        F_np,
        delta_values,
        color="#ff7f0e",
        linewidth=2.5,
        label=f"Δ = grad(pv_expr, 'forward_rate') = {delta_values[0]:.4f}",
    )
    ax_delta.axvline(
        K_val, color="#d62728", linewidth=1.2, linestyle=":", label=f"K = {K_val:.0f}"
    )
    ax_delta.set_xlabel("Preço Forward F (cents/bushel)")
    ax_delta.set_ylabel("∂PV/∂F")
    ax_delta.set_title("Delta via JAX grad — constante para forward linear")
    ax_delta.set_ylim(0.0, 1.1)
    ax_delta.legend()
    ax_delta.grid(True, alpha=0.3)

    plt.tight_layout()
    fig
    return (
        F_np,
        ax_delta,
        ax_pv,
        fig,
        market_scan,
        priced_scan,
        pv_values,
        scan_ids,
        trades_scan,
    )


if __name__ == "__main__":
    app.run()
