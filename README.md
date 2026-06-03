<div align="center">

# Schenberg Risk Toolkit

**Composable, lazy pricing for financial instruments — as a graph of formulas.**

[Concepts](docs/concepts.md) · [Extending](docs/extending.md) · [Examples](examples/)

</div>

Schenberg is a Python pricing toolkit built on lazy [Polars](https://pola.rs)
dataframes, [Pandera](https://pandera.readthedocs.io) boundary schemas, and a
small graph engine (`rustworkx`) for composable pricing formulas. You describe a
price as a DAG of row-local formulas; the engine compiles it into a single lazy
expression and never collects until you ask.

```python
from datetime import date
import polars as pl

from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.api import price_swap

market = MarketSnapshot.from_sources(
    as_of=date(2026, 6, 3),
    sources=[
        MarketSource("curves", pl.DataFrame({
            "id_indexador": [1, 2], "tenor_days": [252, 252],
            "zero_rate": [0.10, 0.05], "forward_rate": [0.12, None],
        }).lazy()),
        MarketSource("fixings", pl.DataFrame({
            "id_indexador": [2], "fixing_date": [date(2026, 6, 3)], "fixing_value": [100.0],
        }).lazy()),
        MarketSource("projected_indexes", pl.DataFrame({
            "id_indexador": [2], "tenor_days": [252], "projected_index": [106.0],
        }).lazy()),
    ],
)

swaps = pl.DataFrame({
    "swap_id": ["SWP-1"], "notional": [1_000_000.0],
    "id_indexador_ativo": [1], "id_indexador_passivo": [2],
    "indexador_kind_ativo": ["CDI"], "indexador_kind_passivo": ["IPCA"],
    "payment_days": [252], "accrual": [1.0], "base_date": [date(2026, 6, 3)],
    "fixed_rate_ativo": [None], "fixed_rate_passivo": [None],
    "real_coupon_ativo": [None], "real_coupon_passivo": [0.02],
}).lazy()

price_swap(swaps, market).collect()   # -> swap_id, npv, ativo_pv, passivo_pv
```

A full, runnable version of this is [`examples/01_price_a_swap.py`](examples/01_price_a_swap.py).

## Why a graph

- **Declarative formulas.** A node's parameter names *are* its dependencies —
  no manual wiring. The engine handles topological order, cycle checks, and a
  shared compile cache.
- **Lazy by construction.** Nothing in the engine calls `.collect()`; a whole
  pricing run is one Polars query you execute once, at the edge.
- **Composable.** `compose()` merges graphs; `Router` dispatches heterogeneous
  instruments; `MarketSnapshot` attaches curves/fixings/FX by declarative joins.
- **Typed at the boundary, fast inside.** Pandera contracts guard the public
  edges; the hot path stays plain Polars expressions.
- **Inspectable.** `dependencies_of`, `required_inputs`, `to_mermaid`, and a
  `stage()` mode that materializes every intermediate for null-propagation
  debugging.

## What's included

- A reusable **formula DAG core** (`ExprGraph`) and a **stage pipeline** (`Pipe`).
- **Swap pricing** for CDI, IPCA, and CPI legs.
- A **generic forward backbone** (`forward_price - strike → future_value →
  present_value → value`) and an **energy forward** composed from it.
- **Portfolio** value, PnL, and DV01 helpers.
- A worked **custom-instrument** example (inflation-linked energy forward)
  showing how to extend the engine.

## Install & run

```bash
uv sync --all-groups        # install (Python 3.12+)
uv run pytest               # unit suite
uv run pytest integration   # integration + performance suite
just check                  # lint + typecheck + test
```

## Documentation

| Doc | What |
|-----|------|
| [docs/concepts.md](docs/concepts.md) | The mental model: `ExprGraph`, `Router`, `MarketSnapshot`, `Pipe`, and the Router-vs-data rule. |
| [docs/extending.md](docs/extending.md) | How to add a custom instrument, index, or payoff variant. |
| [examples/](examples/) | Runnable, self-contained scripts. |

## Project layout

```text
schenberg/
  domain/            Pandera boundary schemas + enums
  core/              ExprGraph, Router, MarketRequirement, Pipe
  market_data/       MarketSnapshot, sources, curve specs, shocks, calendar
  pricing/
    api.py           public pricing facade
    instruments/     swap (cdi/ipca/fixed legs), forward (generic/energy)
    portfolio.py     value / PnL / DV01 helpers
  math/              shared Polars expressions
docs/                concepts + extension guides
examples/            runnable scripts, incl. a custom-instrument package
integration/         end-to-end pipeline + performance tests
containers/          container images (dev image; future CLI image)
plugins/             uv workspace extensions (e.g. schenberg_distributed)
tests/               unit suite + fixtures
```

## Workspace plugins

This repo is a `uv` workspace. Extensions live under `plugins/`. The first,
`schenberg_distributed`, centralizes pricing materialization in execution
contexts (`local` / `ray` / `custom`):

```python
from schenberg_distributed import PricingExecutionContext, collect_pricing

context = PricingExecutionContext.ray(engine="streaming")
result = collect_pricing(lazy_pricing_frame, context=context)
```

## License

MIT.
