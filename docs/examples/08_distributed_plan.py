"""Distributed valuation plan example.

Run with `uv run marimo edit docs/examples/08_distributed_plan.py`.
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """# Distributed plan\n\n"""
        """A whole-node valuation DAG executed locally while preserving Polars laziness."""
    )
    return


@app.cell
def _():
    from datetime import date

    import marimo as mo
    import polars as pl
    from schenberg.market_data.snapshot import MarketSnapshot
    from schenberg.market_data.sources import MarketSource
    from schenberg.pricing import price_forward
    from schenberg_distributed import LocalExecutor, ValuationPlan

    return LocalExecutor, MarketSnapshot, MarketSource, ValuationPlan, date, mo, pl, price_forward


@app.cell
def _(MarketSnapshot, MarketSource, date, pl):
    market = MarketSnapshot.from_sources(
        as_of=date(2026, 6, 6),
        sources=[
            MarketSource(
                "curves",
                pl.DataFrame(
                    {
                        "id_indexador": ["DI"],
                        "tenor_days": [252],
                        "forward_rate": [112.0],
                        "risk_free_rate": [0.10],
                    }
                ).lazy(),
                unique_by=("id_indexador", "tenor_days"),
            )
        ],
    )
    trades = pl.DataFrame(
        {
            "instrument_id": ["FWD-1"],
            "indexer": ["DI"],
            "currency": ["BRL"],
            "strike": [100.0],
            "payment_days": [252],
        }
    ).lazy()
    return market, trades


@app.cell
def _(ValuationPlan, market, price_forward, trades):
    plan = ValuationPlan("forward_distributed_example")
    plan.input("trades", trades)
    plan.input("market", market)

    @plan.node(
        "forward_values",
        trades="trades",
        market="market",
        market_sources=("curves",),
    )
    def forward_values(trades, market):
        return price_forward(trades, market)

    return plan


@app.cell
def _(LocalExecutor, plan):
    executor = LocalExecutor()
    lazy_result = executor.lazy(plan, target="forward_values")
    result = executor.collect(plan, target="forward_values")
    return executor, lazy_result, result


@app.cell(hide_code=True)
def _(mo, plan, result):
    mo.vstack(
        [
            mo.md("## Result"),
            mo.ui.table(result),
            mo.md("## Explain\n```text\n" + plan.explain() + "\n```"),
            mo.md("## Mermaid\n```mermaid\n" + plan.to_mermaid() + "\n```"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
