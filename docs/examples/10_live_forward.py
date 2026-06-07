"""Live forward valuation example.

Run with `uv run marimo edit docs/examples/10_live_forward.py`.
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("# Live forward valuation")
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
    from schenberg_live import LiveValuationEngine, MarketEvent
    from schenberg_viz import to_html

    return (
        LiveValuationEngine,
        LocalExecutor,
        MarketEvent,
        MarketSnapshot,
        MarketSource,
        ValuationPlan,
        date,
        mo,
        pl,
        price_forward,
        to_html,
    )


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
    plan = ValuationPlan("forward_live_example")
    plan.input("trades", trades)
    plan.input("market", market)

    @plan.node("forward_values", trades="trades", market="market", market_sources=("curves",))
    def forward_values(trades, market):
        return price_forward(trades, market)

    return plan


@app.cell
def _(LiveValuationEngine, LocalExecutor, MarketEvent, date, plan):
    engine = LiveValuationEngine(plan=plan, executor=LocalExecutor(), target="forward_values")
    first = engine.on_market_event(
        MarketEvent(source="curves", version="mkt-001", as_of=date(2026, 6, 6))
    )
    second = engine.on_market_event(
        MarketEvent(source="curves", version="mkt-001", as_of=date(2026, 6, 6))
    )
    third = engine.on_market_event(
        MarketEvent(source="curves", version="mkt-002", as_of=date(2026, 6, 6))
    )
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert third.cache_hit is False
    return engine, first, second, third


@app.cell
def _(plan, to_html):
    html = to_html(plan, title="Forward live valuation plan")
    return html


@app.cell(hide_code=True)
def _(first, mo, second, third):
    mo.vstack(
        [
            mo.md("## Cache behavior"),
            mo.md(f"first={first.cache_hit}, second={second.cache_hit}, third={third.cache_hit}"),
            mo.ui.table(first.data),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
