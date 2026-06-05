# Schenberg Risk Toolkit

**Composable, lazy pricing for financial instruments — as a graph of formulas.**

Schenberg is a lazy, contract-oriented pricing DSL built on
[Polars](https://pola.rs) LazyFrames, [Pandera](https://pandera.readthedocs.io)
boundary schemas, and a small graph engine. Inputs, market reads and formulas
are **Terms** inside a **Formula**; the `MarketSnapshot` is the environment
supplied at compute time. The same graph declaration can be interpreted as lazy
Polars, a Mermaid diagram, or explanation text — and never collects until you
ask.

## Five-minute example

```python
from datetime import date
import polars as pl
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.instruments.derivatives.forwards.energy import price_energy_forward

market = MarketSnapshot.from_sources(
    as_of=date(2026, 6, 5),
    sources=[
        MarketSource("energy_forward_curve", pl.DataFrame({
            "submarket": ["SE"], "delivery_period": ["2026-07"],
            "forward_price": [270.0],
        }).lazy()),
        MarketSource("curves", pl.DataFrame({
            "id_indexador": ["PLD"], "tenor_days": [252],
            "risk_free_rate": [0.10],
        }).lazy()),
        MarketSource("fx_rates", pl.DataFrame({
            "currency": ["BRL"], "fx_rate": [1.0],
        }).lazy()),
    ],
)

trades = pl.DataFrame({
    "instrument_id": ["ENG-1"],
    "tenor": [date(2026, 7, 1)],
    "indexer": ["PLD"],
    "currency": ["BRL"],
    "strike": [250.0],
    "payment_days": [252],
    "submarket": ["SE"],
    "incentive": ["I0"],
    "delivery_period": ["2026-07"],
}).lazy()

# Everything is lazy — call .collect() when you want the data.
result = price_energy_forward(trades, market)
print(result.collect().select("instrument_id", "future_value", "present_value", "value"))
```

The same formula also explains itself — no running required:

```python
from schenberg.pricing.instruments.derivatives.forwards.energy.api import energy_forward_formula

print(energy_forward_formula.explain(view="output"))
print(energy_forward_formula.to_mermaid(view="output"))
```

## Where to start

| | |
|---|---|
| **[Examples](examples/index.md)** | Four runnable scripts: generic forward, energy forward, formula introspection, shock scenario — plus a full custom instrument walkthrough. |
| **[Concepts](concepts.md)** | Mental model: Term, Formula, MarketSnapshot, Router, Structure, Workflow, Shock. |
| **[Extending](extending.md)** | Add an instrument, a market dimension, or a convention variant. |
| **[API Reference](api/index.md)** | Auto-generated from source docstrings. |

## Architecture in one diagram

```
domain/        Pandera boundary schemas + contract rules     (no deps)
core/          Term, Formula, FormulaGraph, Router,
               Structure, Fold, Workflow, Shock              (the engine)
market_data/   MarketSnapshot, sources, MarketPath, Shock
pricing/       instruments/derivatives/forwards/…
               (energy forward, generic forward, …)
```

Dependencies flow downward only: `pricing → market_data → core → domain`.
