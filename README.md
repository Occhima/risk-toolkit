<div align="center">

# Schenberg Risk Toolkit

**Lazy pricing DSL built on Polars expressions and typed boundary schemas.**

[Concepts](docs/concepts.md) · [Extending](docs/extending.md) · [Examples](docs/examples/)

</div>

Schenberg is a compact pricing DSL. Inputs, market reads, and formulas are
`Term`s inside a `FormulaGraph`; market data arrives pre-bound via
`MarketSnapshot`, never joined inside the graph. Pricing functions return lazy
frames and do not execute trade-side queries until the caller collects.

## Minimal forward example

```python
from datetime import date
import polars as pl

from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.api import price_forward

market = MarketSnapshot.from_sources(
    as_of=date(2026, 6, 6),
    sources=[
        MarketSource(
            "curves",
            pl.DataFrame({
                "id_indexador": ["DI"],
                "tenor_days": [252],
                "forward_rate": [112.0],
                "risk_free_rate": [0.10],
            }).lazy(),
            unique_by=("id_indexador", "tenor_days"),
        ),
    ],
)

trades = pl.DataFrame({
    "instrument_id": ["FWD-1"],
    "indexer": ["DI"],
    "currency": ["BRL"],
    "strike": [100.0],
    "payment_days": [252],
}).lazy()

result = price_forward(trades, market)  # LazyFrame — still lazy
print(result.collect())
# ┌──────────────┬──────────────┬───────────────┬───────────┐
# │ instrument_id │ future_value │ present_value  │ value     │
# │ str           │ f64          │ f64            │ f64       │
# ╞══════════════╪══════════════╪═══════════════╪═══════════╡
# │ FWD-1         │ 12.0         │ 10.858...      │ 10.858... │
# └──────────────┴──────────────┴───────────────┴───────────┘
```

## Public pricing API

```python
from schenberg.pricing.api import (
    price_forward,               # generic forward
    forward_instrument_value,    # shaped as InstrumentValue for position layer
    price_energy_forward,        # energy forward (energy_forward_curve source)
    ForwardContractPricing,      # input schema (with market mixins)
    EnergyForwardPricing,        # input schema for energy forwards
    forward_formula,             # underlying FormulaGraph (inspectable)
    energy_forward_formula,
)
```

## Position layer

Pricing returns a pure per-instrument `value` (no side, no book). The position
layer lifts it onto a `Position` and converts to the book's reporting currency:

```python
from schenberg.position import position_value, position_risk, book_value_rollup
from schenberg.domain.schemas.position import (
    Position, InstrumentValue, InstrumentRisk, BookContract, ReportingFx,
)

# position_value: exposure, mtm, reported_mtm — one row per position
pv = position_value(positions, value=instrument_values, book=book, fx=fx)

# position_risk: position_delta, position_gamma, ... — one row per position
pr = position_risk(positions, risk=instrument_risk)

# book_value_rollup: aggregate exposure, mtm, reported_mtm to book level
rollup = book_value_rollup.compute(pv)
```

## Design rules

- Pure pricing graphs compute own-currency instrument values only.
- Position/book code owns `side`, `quantity`, book metadata, and reporting FX.
- Market joins are declared as `With[role]` mixins and resolved by `bind`
  before the formula graph runs.
- Missing inputs fail loudly at plan time.
- Market source key uniqueness can be validated once when the snapshot is built.

## Modules

- `schenberg.core` — `FormulaGraph`, `Formula`, `Router`, `Fold`, market
  dependencies, expression IR, and diagnostics.
- `schenberg.market_data` — snapshots, sources, market roles (`With`, `bind`),
  shocks, and market path helpers.
- `schenberg.pricing.api` — the tested forward and energy-forward public pricers.
- `schenberg.position` — position views and measures for exposure, MTM,
  reported MTM, and risk factors.
- `schenberg.domain` — typed boundary schemas (Pandera) for all contract and
  position data.
- `schenberg.math` — Black-Scholes, curve interpolation, and statistics helpers.

## Interactive notebooks

```bash
uv run marimo edit docs/examples/05_soja_forward.py         # soybean forward + delta charts
uv run marimo edit docs/examples/06_position_view.py        # position MTM / exposure / delta
uv run marimo edit docs/examples/07_usdbrl_book_valuation.py  # USD/BRL NDF book + PnL explain
```

## Install and check

```bash
uv sync --all-groups
uv run pytest
uv run poe check    # lint + typecheck + test
```
