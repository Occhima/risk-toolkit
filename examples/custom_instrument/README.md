# Custom instrument: inflation-linked energy forward

This example builds an instrument the library has never heard of, to show how
much you get for free when you extend the engine instead of writing a bespoke
pricer.

> Run it: `uv run python -m examples.custom_instrument.run`

## The instrument

An energy forward whose spread payoff is scaled to nominal terms by an inflation
index (IPCA or CPI), discounted, and converted to the reporting currency:

```
real_spread      = forward_price - strike
inflation_factor = projected_index / base_index
future_value     = real_spread * inflation_factor
present_value    = future_value * discount_factor
value            = present_value * fx_rate
```

## The twist: the index is read at a convention-specific date

The factor isn't read at the tenor date. Each index has its own rule for *which
calendar date* on the inflation curve corresponds to the instrument's tenor:

| Index | `id_indexador` | Reference date for a tenor in year `Y` |
|-------|----------------|----------------------------------------|
| IPCA  | 20             | `Y-01-01` (first day of the year)      |
| CPI   | 10             | `Y-04-01` (April)                      |

So a tenor of **Jun/2029** reads IPCA at **2029-01-01**, while a tenor of
**Jun/2028** reads CPI at **2028-04-01**.

## How the library makes this easy

The work splits into three small pieces, each in its own file:

1. **`conventions.py` — the only index-specific logic.** A registry of
   `InflationConvention` rows and a single Polars expression that turns each
   row's tenor into its `reference_date`. Onboarding a new index is **one line**
   in `CONVENTIONS`; nothing else changes.

2. **`graph.py` — "another graph".** Seven formulas wired by parameter name on the
   same `FormulaGraph` engine the built-ins use, declaring exactly the market data
   it needs via `uses_market(MarketRequirement(...))`. Crucially, the inflation
   curve is joined on `(id_indexador, reference_date)` — the convention date *is*
   the selector.

3. **`pricer.py` — the thin public function.** Normalize (attach
   `reference_date`), run the graph, aggregate. Mirrors the built-in
   `price_energy_forward`. Stays lazy end to end.

## Why no Router here

The pricing **formula is identical** for IPCA and CPI — only a *parameter* (the
reference date) differs, and that parameter is computable as a conditional
column. So the difference is **data, not control flow**, and one graph handles
both. You'd reach for a `Router` only if the indices needed genuinely different
*math* or a different *set* of market sources. (See
[`docs/concepts.md`](../../docs/concepts.md#router-vs-data) for the rule of
thumb.)

## What you got for free

Lazy execution, the declarative market join, null-propagation debugging via
`inflation_energy_graph.stage(...)`, dependency introspection
(`.dependencies_of(...)`, `.to_mermaid()`), and a clean place for the next index
or the next payoff variant — without touching the engine.
