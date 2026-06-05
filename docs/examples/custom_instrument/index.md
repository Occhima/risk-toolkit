# Custom instrument: inflation-linked energy forward

**Run it:** `uv run python -m docs.examples.custom_instrument.run`

This example builds an instrument the library has never heard of, to show how
much you get for free when you extend the engine instead of writing a bespoke
pricer.

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

A tenor of **Jun/2029** reads IPCA at **2029-01-01**, while a tenor of
**Jun/2028** reads CPI at **2028-04-01**.

## How the library makes this easy

The work splits into three small files:

1. **[`conventions.py`](conventions.py) — the only index-specific logic.** A
   registry of `InflationConvention` rows and a Polars expression that turns each
   tenor into its `reference_date`. Onboarding a new index is **one line** in
   `CONVENTIONS`; nothing else changes.

2. **[`graph.py`](graph.py) — "another formula".** Seven formulas wired with
   `uses(term)` on the same `Formula` engine the built-ins use, declaring exactly
   the market data it needs via `MarketRequirements`. The inflation curve is joined
   on `(id_indexador, reference_date)` — the convention date *is* the selector.

3. **[`pricer.py`](pricer.py) — the thin public function.** Normalize (attach
   `reference_date`), run the formula, aggregate. Stays lazy end to end.

## Why no Router here

The pricing **formula is identical** for IPCA and CPI — only a *parameter* (the
reference date) differs, and that parameter is computable as a conditional column.
So the difference is **data, not control flow**, and one formula handles both.
You'd reach for a `Router` only if the indices needed genuinely different *math*
or a different *set* of market sources. See
[Concepts → Router vs data](../../concepts.md#router-vs-data) for the rule.

## What you get for free

- Lazy execution end to end
- Declarative market joins (no manual `.join()`)
- Null-propagation debugging via `inflation_energy_graph.stage(...)`
- Dependency introspection: `.dependencies_of(...)`, `.to_mermaid()`
- A clean place for the next index convention — one line in `CONVENTIONS`
