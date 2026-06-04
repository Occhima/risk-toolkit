# Simplifying Schenberg — what changed and why

A pro-quant read of the codebase turned into three landed refactors. This is the
record of what moved, the rationale, and the one judgment call I made against my
own first instinct. **All of it is behaviour-preserving: the full suite stays
green and every priced number is unchanged.** The companion
[`debugging_capabilities.ipynb`](./debugging_capabilities.ipynb) shows the shared
backbone in a picture.

The library was already well-factored (`discount_graph` shared, a crisp
Router/data split, honest boundary contracts), so these remove the *last* layer of
repetition rather than rescuing a mess.

---

## 1. One discounted-cashflow backbone for forwards and swap legs

**The observation.** Put `forward_valuation` next to `swap_leg_valuation` and they
are the same machine — both end in `future_value × discount_factor`:

```
forward:   forward_price - strike   ─► future_value ─►(× DF)─► present_value ─►(× fx)─► value
swap leg:  cashflow_amount × sign    ─► future_value ─►(× DF)─► present_value
```

**What changed.** The `present_value = future_value × discount_factor` step now
lives once, in `discounted_cashflow_graph` (`schenberg/pricing/discounting.py`).
Each instrument composes it and supplies only the payoff that defines
`future_value`:

- the forward payoff is `future_value = forward_price - strike`, plus its FX step;
- the swap-leg payoff is `future_value = cashflow_amount × pay_receive_sign` (its
  signed cashflow), with no FX step.

`LegPricing` keeps its public `signed_cashflow` and `pv` columns through view
overrides (`signed_cashflow → future_value`, `pv → present_value`), so the
contract and the numbers are untouched. The duplicated discount-and-multiply node
is gone, and "discount a future cashflow" is stated exactly once.

The pay/receive sign is the leg's signed **quantity** — the same role `side` plays
for a structured-product leg, which is the conceptual unification the original
hypothesis was after.

### The judgment call I made (and why)

My first note said to *delete `aggregate_swap_pv` and route swaps through
`price_structures`*. I did **not** do that, on purpose. `price_swap` returns a
genuinely richer contract than `price_structures` — `npv` **plus** the signed
`ativo_pv` / `passivo_pv` breakdown — and that breakdown is pinned by tests, the
example, and the README. Collapsing swaps into the generic `InstrumentPrice`
(single `price`) would *delete a useful, tested capability*, which is destruction,
not simplification. So I unified the **valuation backbone** (the real duplication)
and left the swap's richer aggregation intact. This matches my own caveat that the
full collapse is breaking and belongs behind a version bump.

---

## 2. `compose` carries views + a single `assemble` verb

**The wart.** `FormulaGraph.compose` used to *drop* declared views, so every graph
that set a view had to re-declare it after each compose — the
`compose → for_market → returns` triple was spelled out by hand in
`swap/generic.py`, `forward/generic.py`, `forward/energy.py`, and `register_leg`.

**What changed.**

- `compose` now carries views through composition (conflicting view columns are a
  hard error), so the redundant second `.returns(...)` is gone everywhere.
- `FormulaGraph.assemble(name, *graphs, market=, fixed_market=, schema=, view=)`
  is the single instrument-assembly verb. Every instrument graph — swap legs via
  `register_leg`, the base and energy forwards — is built through it instead of
  re-spelling the recipe.
- The dead `swap/engine.py` back-compat shim is deleted; the docs point at the live
  modules.

---

## 3. A simpler market-data layer

**The duplication.** `CurveSpec`, `FxRatesSpec`, `FixingsSpec` and the DI curve
each hand-rolled the same `MarketRequirement(table=…, on=ColumnSet.from_pairs(…),
outputs={value: output})` block, and a separate `MarketRead` class existed purely
to defer naming the output column — forcing a two-`@overload` signature onto every
spec method.

**What changed.**

- One `JoinSpec.read(...)` (`schenberg/market_data/specs.py`) builds every keyed
  read; the named specs are now thin wrappers that just pin a table and the
  semantic key names.
- `MarketRead` is **deleted**. A requirement's output simply defaults to the value
  column's name, and `FormulaGraph.for_market(rate=…)` renames it to the keyword
  via a uniform `MarketDependency.with_output`. The overloads and the
  "use uses_market" mismatch error are gone — `for_market` just names the column,
  as its docstring always claimed.
- Genuinely different reads stay separate behind the same `MarketDependency`
  protocol: interpolated reads (the vol surface) and the multi-output energy
  forward curve are untouched in behaviour.

---

## What was deliberately left alone

- **The hot path.** Every change is in *declaration*; the engine still compiles to
  one lazy `with_columns`. No per-node validation crept in.
- **Options.** Their payoff genuinely forks (call vs put, model vs model) — that is
  what the `Router` is for, so they are *not* folded into the linear
  discounted-cashflow family.
- **The swap NPV breakdown.** See §1 — kept on purpose.
