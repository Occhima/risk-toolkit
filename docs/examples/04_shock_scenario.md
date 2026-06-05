# 04 — Shock scenario

**Run it:** `uv run python docs/examples/04_shock_scenario.py`

A `Shock` is an **endomorphism** `MarketSnapshot → MarketSnapshot`: it returns a
*new* snapshot with sources transformed, never mutating the original. Shocks
compose associatively — you can build a multi-factor stress by chaining them.

## Key properties

- **Immutability**: `market.apply(shock)` returns a new snapshot. The original
  `market` is unchanged. You can price base and stressed scenarios from the same
  object without copying data.
- **Composition**: `Shock.compose(bump_rates, bump_vol, ...)` chains multiple
  shocks; order matters but the composition is explicit.
- **Explainability**: `shock.explain()` / `shock.info()` describe what was bumped.

## Two equivalent ways to build a +100bp parallel shift

```python
# MarketPath: lens-lite access into a specific source column
bump = MarketPath("curves").column("risk_free_rate").modify(lambda r: r + 0.01)

# curve_parallel_shift: canned helper
same_bump = curve_parallel_shift(source="curves", shift=0.01, column="risk_free_rate")
```

Both produce the same `Shock`. Use `MarketPath` for arbitrary transforms; use
`curve_parallel_shift` for the common parallel-bump case.

## Source

```python
--8<-- "docs/examples/04_shock_scenario.py"
```
