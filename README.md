# Schenberg Risk Toolkit

Schenberg is a Python risk-pricing toolkit built around lazy Polars dataframes,
Pandera boundary schemas, and a small graph engine for composable pricing
formulae.

## What is included

- A reusable formula DAG core backed by `rustworkx`.
- Declarative market-data attachment via `MarketSnapshot` requirements.
- Swap pricing for CDI, IPCA, and CPI legs.
- A generic forward valuation backbone:
  `future_value -> present_value -> value`.
- Energy forward pricing composed from the generic forward backbone.
- Portfolio value, PnL, and DV01 helper pipelines.
- Ruff, ty, pytest, Poe, direnv, and Nix/Flake-ready project configuration.

## Project layout

```text
schenberg/
  core/                         # graph engine, market joins, pipeline object
  domain/                       # Pandera public schemas
  pricing/
    api.py                      # public pricing facade
    instruments/
      swap/                     # swap transforms and graph engine
      forward/
        generic.py              # generic forward valuation graph
        energy.py               # energy forward pricer
      options.py                # Black-Scholes example graph
    portfolio.py                # value / PnL / risk helpers
tests/                          # functional pytest suite and conftest fixtures
.agents/                        # notes for agentic engineering harnesses
```

## Quick start

```bash
uv sync --all-groups
uv run pytest
```

If you use direnv with Nix flakes:

```bash
direnv allow
just check
```

## Minimal pricing example

```python
from datetime import date

import polars as pl

from schenberg.core.market import MarketSnapshot
from schenberg.pricing.api import price_swap

swaps = pl.DataFrame(...).lazy()
market = MarketSnapshot(as_of=date(2026, 6, 3), curves=pl.DataFrame(...).lazy())
result = price_swap(swaps, market).collect()
```

## Development commands

The project exposes both Poe tasks in `pyproject.toml` and `just` shortcuts:

```bash
uv run ruff check .
uv run ty check
uv run pytest
```

or:

```bash
just check
```
