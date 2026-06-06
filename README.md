<div align="center">

# Schenberg Risk Toolkit

**Small lazy pricing over Polars expressions.**

[Concepts](docs/concepts.md) · [Extending](docs/extending.md) · [Examples](docs/examples/)

</div>

Schenberg is a compact pricing DSL built on lazy [Polars](https://pola.rs)
dataframes and typed boundary schemas. Inputs, market reads, and formulas are
`Term`s inside a `FormulaGraph`; `MarketSnapshot` supplies market sources at
compute time. Pricing functions return lazy frames and do not execute trade-side
queries until the caller collects the result.

The implemented public pricing surface is intentionally small:

```python
from schenberg.pricing.api import (
    price_forward,
    forward_instrument_value,
    price_energy_forward,
    energy_forward_instrument_value,
)
```

Current public pricers cover generic forwards and energy forwards. The position
layer can then lift own-currency `InstrumentValue` rows onto positions/books and
apply reporting FX conversion.

## Minimal forward example

```python
from datetime import date
import polars as pl

from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.api import price_forward

market = MarketSnapshot.from_sources(
    as_of=date(2026, 6, 5),
    sources=[
        MarketSource(
            "curves",
            pl.DataFrame(
                {
                    "id_indexador": ["IDX"],
                    "tenor_days": [252],
                    "forward_rate": [110.0],
                    "risk_free_rate": [0.10],
                }
            ).lazy(),
            unique_by=("id_indexador", "tenor_days"),
        ),
    ],
)

trades = pl.DataFrame(
    {
        "instrument_id": ["FWD-1"],
        "tenor": [date(2027, 6, 5)],
        "indexer": ["IDX"],
        "currency": ["USD"],
        "strike": [100.0],
        "payment_days": [252],
    }
).lazy()

result = price_forward(trades, market)  # LazyFrame
print(result.collect())
```

## Design rules

- Pure pricing graphs compute own-currency instrument values only.
- Position/book code owns `side`, `quantity`, book metadata, and reporting FX.
- Market joins are declared as requirements and attached at compute time.
- Missing graph inputs fail loudly; staged debugging can opt into null columns.
- Market source key uniqueness can be validated once when the snapshot is built.
- Interpolation may precompute a quote grid, but trade-side work remains lazy
  after the interpolation book has been built.

## Included modules

- `schenberg.core`: `FormulaGraph`, `Formula`, `Router`, `Fold`, market
  dependencies, and diagnostics.
- `schenberg.market_data`: market snapshots, keyed/interpolated requirements,
  market source validation, shocks, and market object helpers.
- `schenberg.pricing.api`: the tested forward and energy-forward public pricers.
- `schenberg.position`: position views and measures for exposure, MTM, and
  reporting-currency conversion.

## Install and check

```bash
uv sync --all-groups
uv run pytest
uv run poe check
```
