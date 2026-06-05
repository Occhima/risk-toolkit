"""An energy-forward pricing graph, explored as a marimo notebook.

Run it live:   uv run marimo edit examples/notebooks/energy_forward.py
Or read-only:  uv run marimo run  examples/notebooks/energy_forward.py

The notebook walks the typed, contract-oriented DSL end to end: the trade
contract, the market *requirements* (every join declared once), the pure formula
graph, and a lazy Polars plan that prices a book -- with a live strike slider.
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # ⚡ Energy forward — a contract-oriented pricing graph

        A Schenberg instrument is three typed schemas and some pure formulas:

        * a **Contract** — what each trade row carries,
        * **MarketRequirements** — *what market data it needs and how to find each row*
          (every join is declared here, and nowhere else),
        * an **Output** contract — what the priced book looks like.

        Formulas read `g.contract` and `g.market` terms and **never join**.
        `g.bind(trades, market)` resolves the market; `g.plan(bound)` returns one
        lazy Polars plan.
        """
    )
    return


@app.cell
def _():
    from datetime import date

    import polars as pl

    from schenberg.market_data.snapshot import MarketSnapshot
    from schenberg.market_data.sources import MarketSource
    from schenberg.pricing.instruments.forward.energy import (
        EnergyForwardRequirements,
        energy_forward_graph,
        price_energy_forward,
    )

    return (
        EnergyForwardRequirements,
        MarketSnapshot,
        MarketSource,
        date,
        energy_forward_graph,
        pl,
        price_energy_forward,
    )


@app.cell
def _(EnergyForwardRequirements, mo):
    requirements = {
        field: f"{dep.table}({', '.join(dep.left_keys)})"
        for field, dep in EnergyForwardRequirements.__requirements__.items()
    }
    mo.md(
        "## Market requirements\n\n"
        "Each field names a market column the instrument exposes and the keyed read "
        "that fills it. `.by(...)` is omitted here — the reads use their typed default "
        "keys.\n\n" + "\n".join(f"- **{field}** ⟵ `{src}`" for field, src in requirements.items())
    )
    return


@app.cell
def _(energy_forward_graph, mo):
    mo.md(f"## The pricing graph\n\n```\n{energy_forward_graph.explain()}\n```")
    return


@app.cell
def _(energy_forward_graph, mo):
    mo.md("## Dependency graph").center()
    return


@app.cell
def _(energy_forward_graph, mo):
    mo.mermaid(energy_forward_graph.to_mermaid())
    return


@app.cell
def _(mo):
    strike = mo.ui.slider(start=80.0, stop=160.0, step=5.0, value=100.0, label="Strike (R$/MWh)")
    strike
    return (strike,)


@app.cell
def _(MarketSnapshot, MarketSource, date, pl):
    market = MarketSnapshot.from_sources(
        as_of=date(2026, 6, 3),
        sources=[
            MarketSource(
                "energy_forward_curve",
                pl.DataFrame(
                    {
                        "submarket": ["SE", "SE"],
                        "delivery_period": ["2026-07", "2026-08"],
                        "forward_price": [120.0, 130.0],
                        "settle_days": [30, 60],
                    }
                ).lazy(),
            ),
            MarketSource(
                "di_curve",
                pl.DataFrame(
                    {
                        "curve_name": ["DI", "DI"],
                        "id_indexador": [1, 1],
                        "tenor_days": [30, 60],
                        "zero_rate": [0.10, 0.10],
                    }
                ).lazy(),
            ),
            MarketSource(
                "fx_rates",
                pl.DataFrame({"currency": ["BRL"], "fx_rate": [1.0]}).lazy(),
            ),
        ],
    )
    return (market,)


@app.cell
def _(date, pl, strike):
    # One instrument (ENG-1) delivering across two monthly periods -> two legs.
    legs = pl.DataFrame(
        {
            "instrument_id": ["ENG-1", "ENG-1"],
            "instrument_type": ["FORWARD", "FORWARD"],
            "forward_family": ["ENERGY", "ENERGY"],
            "settlement_type": ["PHYSICAL", "PHYSICAL"],
            "submarket": ["SE", "SE"],
            "delivery_period": ["2026-07", "2026-08"],
            "id_indexador": [1, 1],
            "payment_days": [30, 60],
            "strike": [strike.value, strike.value],
            "currency": ["BRL", "BRL"],
            "fixing_date": [date(2026, 8, 6), date(2026, 9, 4)],
        }
    ).lazy()
    return (legs,)


@app.cell
def _(legs, market, mo, price_energy_forward):
    priced = price_energy_forward(legs, market).collect()
    mo.vstack(
        [
            mo.md(f"## Priced book — **R$ {priced['price'][0]:,.2f}**"),
            mo.ui.table(priced, selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
