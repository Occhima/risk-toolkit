<div align="center">

# Schenberg Risk Toolkit

**Composable, lazy pricing for financial instruments — as a graph of formulas.**

[Concepts](docs/concepts.md) · [Extending](docs/extending.md) · [Examples](examples/)

</div>

Schenberg is a lazy, contract-oriented pricing DSL built on lazy
[Polars](https://pola.rs) dataframes, [Pandera](https://pandera.readthedocs.io)
boundary schemas, and a small graph engine (`rustworkx`). Inputs, market reads
and formulas are **Terms** inside a **FormulaGraph**; the `MarketSnapshot` is the
environment supplied at compute time. The same declaration can be interpreted as
lazy Polars, a Mermaid diagram, explanation text, or debug stages, and never
collects until you ask. A `Router` is a contract-oriented choice among pricing
graphs; a **Structure** composes pure component pricing with exposure and a
**Fold** (so position direction lives outside the pricing math); a **Shock** is an
endomorphism on the market for scenarios; a `Workflow` handles shape-changing
dataframe stages.

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

# A swap *is* its legs — booked directly as normalized SwapLegInput rows.
# leg_weight (+1 receive / -1 pay) is position direction: it belongs to the swap
# Structure's exposure, never to the pure leg-pricing graph.
_common = {
    "notional": 1_000_000.0, "payment_days": 252, "accrual": 1.0,
    "base_date": date(2026, 6, 3), "fixed_rate": None, "cashflow_amount": None,
}
legs = pl.DataFrame([
    {"swap_id": "SWP-1", "leg_id": "ativo", "leg_kind": "CDI",
     "leg_role": "ativo", "leg_weight": 1.0, "id_indexador": 1, "real_coupon": None, **_common},
    {"swap_id": "SWP-1", "leg_id": "passivo", "leg_kind": "IPCA",
     "leg_role": "passivo", "leg_weight": -1.0, "id_indexador": 2, "real_coupon": 0.02, **_common},
]).lazy()

price_swap(legs, market).collect()   # -> swap_id, npv, ativo_pv, passivo_pv
```

A full, runnable version of this is [`examples/01_price_a_swap.py`](examples/01_price_a_swap.py).

## Why a graph

- **Terms, explicitly wired.** A formula names its dependencies with `uses(term)`;
  the engine handles topological order, cycle checks, and a shared compile cache.
- **Lazy by construction.** Nothing in the engine calls `.collect()`; a whole
  pricing run is one Polars query you execute once, at the edge.
- **Composable.** `merge` / `extend` / `then` compose graphs as open graphs;
  `Router` is a contract-oriented choice among them; `Structure` + `Fold` compose
  weighted components; `MarketSnapshot` is the environment that supplies
  curves/fixings/FX/vol by declarative, lookup-oriented reads.
- **Direction stays out of the math.** Pure pricing graphs never see
  `side`/`pay_receive`; weighting and aggregation live in a `Structure`.
- **Typed at the boundary, fast inside.** Pandera contracts guard the public
  edges; the hot path stays plain Polars expressions.
- **One declaration, many interpretations.** `explain`, `info`, `to_mermaid`, and
  a `stage()` mode that materializes every intermediate for null-propagation
  debugging — all derived from the same graph, so they can't drift.

## What's included

- A reusable **formula DAG core** (`FormulaGraph`), a **Structure**/`Fold` layer
  for weighted components, and a **stage pipeline** (`Workflow`).
- **Swap pricing** for CDI, IPCA, and CPI legs — pure leg pricing folded by a swap
  `Structure` that applies `leg_weight`.
- **Option pricing** under generalized Black-Scholes-Merton (GENERALIZED and
  MERTON), priced off an interpolated **volatility surface**, with **Greeks
  three ways** — closed-form, finite-difference, and autograd — that reconcile.
- A **generic forward backbone** (`forward_price - strike → future_value →
  present_value → value`) and an **energy forward** composed from it.
- A **position layer** (`PositionView`): pure instrument value × position × book
  context → measures (`exposure`, `position_notional`, `mtm`, `reported_mtm`, and
  an additive **PnL explain**), declared like pricing formulas and rolled up to
  books with a `Fold`.
- **Scenarios** via `Shock` (endomorphism on `MarketSnapshot`) and `MarketPath`
  (a lens-lite onto a source/column).
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
| [docs/concepts.md](docs/concepts.md) | The mental model: `Term`, `FormulaGraph`, `Router` as ArrowChoice, `Structure` + `Fold`, `MarketSnapshot` as the Reader environment, `Shock`/`MarketPath`, `Workflow`, and the Router-vs-data rule. |
| [docs/extending.md](docs/extending.md) | How to add a custom instrument, index, or payoff variant. |
| [examples/](examples/) | Runnable, self-contained scripts. |

## Project layout

```text
schenberg/
  domain/            Pandera boundary schemas + enums
  core/              Term, FormulaGraph, Router, Structure, Fold, MarketRead, Workflow, diagnostics
  market_data/       MarketSnapshot, sources, curve specs, MarketPath, Shock, calendar
  pricing/
    api.py           public pricing facade
    instruments/     swap (legs + structure), forward (generic/energy), option
    structured.py    structured products as weighted sums of components
  position/          PositionView, reusable measures, built-in value/PnL views
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
