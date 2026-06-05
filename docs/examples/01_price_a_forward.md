# 01 — Price a generic forward

**Run it:** `uv run python docs/examples/01_price_a_forward.py`

The simplest entry point. A forward pays `forward_price − strike` at tenor,
discounted at the risk-free rate and converted to the reporting currency.

## What to notice

**Contract rules** — Two columns are derived automatically from the contract
terms before Pandera validates the input. You never call `.resolve()` yourself:

- `index_fixing_date`: the date the index is observed. Default → same as tenor;
  `"CPI"` shifts it +5 calendar days.
- `currency_fixing_date`: the date the FX rate is observed. Default → same as
  tenor; `"EUR"` snaps to the previous business day.

Omit either column and it will be filled for you. Supply it and it is preserved
as-is.

**Market structure:**

- `curves` keyed by `(id_indexador, tenor_days)` → supplies `forward_rate` and
  `risk_free_rate`.
- `fx_rates` keyed by `currency` → supplies the FX conversion factor.

The join key columns on the *contract* side are `indexer` and `payment_days`
(the `ForwardMarket` requirements map them to the curve's `id_indexador` and
`tenor_days` via `.by(...)`).

## The formula (unchanged for all forwards)

```
year_fraction   T  = payment_days / 252
discount_factor DF = exp(−risk_free × T)
future_value    FV = forward_price − strike
present_value   PV = FV × DF
value           V  = PV × currency
```

## Source

```python
--8<-- "docs/examples/01_price_a_forward.py"
```
