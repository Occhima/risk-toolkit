# Where Schenberg can still be simplified

A pro-quant read of the codebase, written as a companion to
[`debugging_capabilities.ipynb`](./debugging_capabilities.ipynb). The notebook
makes the case visually; this document turns it into concrete, line-counted
proposals. Each one is rated **safe / moderate / speculative** by how much of the
public surface it disturbs.

The library is already well-factored — `discount_graph` is shared, the Router/data
distinction is crisp, the boundary contracts are honest. So these are about
removing the *last* layer of repetition, not rescuing a mess.

---

## 1. A swap leg **is** a routed structured forward  *(speculative — the big one)*

This is the hypothesis from the prompt, and the graphs confirm it. Put
`forward_valuation` and `swap_leg_valuation` side by side
([`images/03_forward_vs_swap.png`](./images/03_forward_vs_swap.png)) and they are
the *same machine*:

```
forward:   forward_price - strike  ─► future_value  ─►(× DF)─► present_value ─►(× fx)─► value
swap leg:  cashflow_amount × sign   ─► signed_cashflow ─►(× DF)─►        pv
```

Both compose `discount_graph` (`year_fraction → discount_factor`) and both end in
**`<a future cashflow> × discount_factor`**. The differences are cosmetic:

| | forward | swap leg |
|---|---|---|
| payoff node | `future_value = forward_price − strike` | `signed_cashflow = cashflow_amount × pay_receive_sign` |
| FX step | yes (`value`) | no |
| output schema | `ForwardPricing` | `LegPricing` |

And the **aggregation rhymes too**. `aggregate_swap_pv` sums signed leg PVs per
`swap_id`:

```python
priced_legs.group_by("swap_id").agg(npv=pv.sum(), ...)   # pv already signed
```

…which is the *exact* shape `price_structures` already implements generically —
`side × quantity × price` summed per `structure_id`. **A swap is a two-leg
structured product whose `quantity` is the pay/receive sign (±1).**

### The unification

Introduce one `discounted_cashflow` backbone (the thing forwards and legs already
half-share) and express each instrument as **payoff + (optional) fx**:

```python
# schenberg/pricing/discounted_cashflow.py  (new)
def discounted_instrument(name, payoff, *, fx=False, schema, market):
    parts = [discount_graph, payoff] + ([fx_payoff] if fx else [])
    return FormulaGraph.compose(name, *parts).for_market(**market).returns("pricing", schema)
```

Then:
- the forward payoff is `future_value = forward_price - strike` with `fx=True`;
- every swap leg payoff ends in `signed_cashflow`, `fx=False`;
- `aggregate_swap_pv` is deleted and swaps route through `price_structures` with
  `quantity = pay_receive_sign`.

**Payoff.** One backbone, one aggregator, for the whole discounted-cashflow family
(forwards, all swap legs, and any future linear instrument).

**Why speculative, not safe.** It collapses two public schemas (`LegPricing`,
`ForwardPricing`) and two public functions (`price_swap`, `aggregate_swap_pv`)
into the structured-product path. That is a breaking API change and wants a
deprecation window. Do **not** fold options in here — their payoff genuinely forks
(that is what the `Router` is for). The line is: *linear, discount-a-cashflow
instruments unify; anything with optionality stays separate.*

---

## 2. Kill the `compose → for_market → returns` boilerplate  *(safe — do this first)*

The same three-line incantation is spelled out by hand in at least three places
(`swap/generic.py`, `forward/generic.py`, and inside `register_leg`):

```python
base_forward_graph = (
    FormulaGraph.compose("base_forward", forward_valuation_graph)
    .uses_market(DI.zero_rate(), FX.fx_rate())
    .returns("pricing", ForwardPricing)
)
```

Worse, **`returns` has to be repeated** because `FormulaGraph.compose` silently
*drops* views — `swap_leg_valuation_graph` sets `.returns("pricing", LegPricing)`,
and then `base_swap_leg` composes it and has to set the identical view *again*.
That is a real wart, visible in `core/graph.py::compose` (it copies formulas,
aliases and market, but never `_views`).

Two cheap fixes, independent of proposal 1:

1. **Carry views through `compose`** when they don't conflict. One added loop in
   `compose`; removes every redundant second `.returns(...)`.
2. **Generalise `register_leg` into the one assembly verb** for *all* instruments,
   not just swap legs:

   ```python
   def assemble(name, *graphs, market, schema, view="pricing"):
       return FormulaGraph.compose(name, *graphs).for_market(**market).returns(view, schema)
   ```

   `register_leg` already proves the pattern works; it's just scoped to legs. Lift
   it to `core/` and the three ad-hoc assemblies become one call each.

**Payoff.** Removes ~3 repeated blocks and one whole class of bug (a view that
exists on the inner graph but was forgotten on the composed one). No public API
change.

---

## 3. One `JoinSpec` instead of four near-identical specs  *(moderate)*

`CurveSpec`, `FxRatesSpec`, `FixingsSpec` (and the DI/vol specs) are four frozen
dataclasses that each return a `MarketRequirement` differing only in **(table,
key columns, value column)**:

```python
# CurveSpec.value, FxRatesSpec.fx_rate, FixingsSpec.fixing — all the same shape:
MarketRequirement(
    table=self.name,
    on=ColumnSet.from_pairs((left, right), ...),
    outputs={value_col: output},
)
```

This is begging to be one parametric spec:

```python
@dataclass(frozen=True, slots=True)
class JoinSpec:
    table: str
    keys: tuple[tuple[str, str], ...]      # (left_default, right) pairs
    def read(self, value_col, *, output=None, **key_overrides) -> MarketRequirement | MarketRead:
        ...
```

`CurveSpec("curves")` becomes `JoinSpec("curves", keys=(("id_indexador","id_indexador"),
("payment_days","tenor_days")))` — a one-liner instead of a 50-line file. Interpolated
reads (the vol surface) stay a *separate strategy* behind the existing
`MarketDependency` protocol; only the plain-join specs collapse.

### …and delete the `MarketRead` closure while you're there

`MarketRead` exists only to defer naming the output column, and it does it with a
`Callable[[str], Any]` builder closure plus an `@overload` dance in every spec.
But `MarketRequirement` is a frozen dataclass — the deferral is just a field that
isn't set yet. `for_market` can finalise it with `dataclasses.replace`:

```python
# core/graph.for_market, conceptually:
req = read if isinstance(read, MarketRequirement) else replace(read, outputs={read.value: output})
```

That removes the `MarketRead` class, the `build` closure, and the two-overload
signature from every spec — pure subtraction.

**Payoff.** Four spec files → one + thin aliases; one fewer indirection class; the
overloads disappear. **Risk:** `MarketRequirement.attach` and `ColumnSet` are the
load-bearing join; this touches them, so it needs the existing market-data tests
green before and after (they are good, so this is mechanical, not scary).

---

## Suggested order

1. **#2** (safe, no API change) — immediate cleanup, and `assemble`/view-carrying
   make the next two easier.
2. **#3** (moderate, internal) — collapses the market layer behind the same public
   `for_market` calls.
3. **#1** (speculative, breaking) — the real prize, but it needs a deprecation
   path for `price_swap`/`LegPricing`. Land it last, behind a version bump.

None of these touch the hot path: the engine still compiles to one lazy
`with_columns`. They remove *declaration* repetition, not execution machinery —
which is exactly the kind of simplification that keeps a young library from
ossifying.
