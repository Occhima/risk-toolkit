# 02 — Energy forward

**Run it:** `uv run python docs/examples/02_energy_forward.py`

The energy forward reuses the *same* formula as the generic forward. The only
differences are:

1. `forward_price` comes from `energy_forward_curve` keyed by
   `(submarket, delivery_period)` instead of a rates curve.
2. The `EnergyForwardPricing` contract adds `submarket`, `incentive`, and
   `delivery_period` columns — routing/selection coordinates, not new math.
3. The **PLD** indexer overrides the default fixing-date rule: its
   `index_fixing_date` is the **6th business day of the month after delivery**.

## Contract specialisation without new formulas

```
EnergyForwardPricing
  ├── ForwardContractPricing      ← core contract + mixin rules
  │     ├── IndexerFixingMixin    ← index_fixing_date rule (default: same_day)
  │     ├── CurrencyFixingMixin   ← currency_fixing_date rule
  │     └── TenorMixin            ← tenor + payment_days
  └── [override] PLD rule         ← 6th biz-day of following month
```

Adding a new indexer convention is one `@rule_for` override on a subclass.
The formula graph never needs to change.

## Source

```python
--8<-- "docs/examples/02_energy_forward.py"
```
