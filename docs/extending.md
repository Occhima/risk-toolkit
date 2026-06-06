# Extending Schenberg

Keep extensions small and lazy.

## Add a formula

1. Define or reuse a Pandera dataframe model for contract inputs.
2. Define `MarketRequirements[Contract]` with the market columns needed by the
   formula.
3. Build a `Formula`/`FormulaGraph` with pure `pl.Expr` terms.
4. Expose a thin public function only when it is tested.

Pricing graph code should wire terms and compose expression helpers. Numerical
math belongs in `schenberg.math`; graph builders should not hide heavy formulas.

## Reuse forward math

Forward-like instruments should reuse the public builder:

```python
from schenberg.pricing.instruments.derivatives.forwards.formulas import build_forward_formula

my_forward_formula = build_forward_formula(
    name="my_forward",
    contract=MyForwardContract,
    market=MyForwardMarket,
)
```

The builder computes `future_value`, `present_value`, and `value` in the
contract's own currency. Reporting FX conversion belongs in the position layer.

## Market data

Use `MarketSource(..., unique_by=(...))` for quote tables whose keys must be
unique. Snapshot construction may collect those market sources to validate quote
metadata. Avoid `.collect()` in pricing or position code.

Interpolated requirements may precompute quote grids. Keep that boundary explicit
in docs/tests and keep the returned trade frame lazy.

## Public API discipline

Do not document a public pricer until it is importable and tested from
`schenberg.pricing.api`. If an example imports a symbol, that symbol must exist.
