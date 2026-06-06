# Concepts

Schenberg is a **lazy, contract-oriented pricing DSL**. A formula is a statically
inspectable pricing program: its inputs, market reads, and derived values are all
**Terms** wired into a **FormulaGraph**. The same graph declaration is *interpreted*
many ways — compiled to lazy Polars, rendered as a Mermaid diagram, explained as
text, or reported as market requirements. `collect()` is yours to call, once, at
the edge.

## The building blocks

| Block | What it is | When you reach for it |
|-------|------------|-----------------------|
| **`Term`** | One value in a graph: an input, a market read, a literal, or a formula. | The atom everything else is made of. |
| **`FormulaGraph`** | An open, typed, applicative graph of terms that compiles to one nested `pl.Expr`. | Pure, row-local math where the shape doesn't change: discounting, payoffs, factors. |
| **`MarketSnapshot`** | The Reader *environment*: named market sources supplied at compute time. | Looking up curves/fixings/FX/vol onto trade rows. |
| **`Router`** | A contract-oriented case split over computations (ArrowChoice). | The *formula* differs per row (instrument family, option kind, ...). |
| **`Structure`** | Component pricing → exposure/weighting → `Fold`. | Structured instruments (a swap *is* its legs): the layer that owns position direction. |
| **`Fold`** | Monoidal aggregation of component rows into instrument/portfolio rows. | "Group by id and combine values" — sums, weighted sums, filtered sums. |
| **`Shock` / `MarketPath`** | An endomorphism `MarketSnapshot → MarketSnapshot`, focused by a lens-lite path. | Scenarios and risk: bump a curve/vol, reprice, compose stresses. |
| **`Workflow`** | A DAG of *stages* returning LazyFrames. | Steps that change shape: joins, group-bys, repricing under a bumped market. |

## 1. Terms

Everything in a graph is a `Term`. There are four kinds you work with:

- **Input terms** — boundary columns the caller supplies. They come from the
  graph's input schema: `g.input.spot`. Accessing an undeclared column raises.
- **Market terms** — boundary columns an attached market read supplies:
  `m.rate`, `m.vol`. Their *name is the output column* the read writes.
- **Formula terms** — values derived from other terms. The `@g.formula` decorator
  registers one and **returns it**, so later formulas can depend on it.
- **View terms** — the terms a `returns(...)` view publishes as result columns.

A term carries its `name`, `kind`, and pure-introspection metadata (`symbol`,
`latex`, `description`). It is what you pass to `uses(...)` and `returns(...)`.

## 2. FormulaGraph

A formula names its dependencies explicitly with `uses(term)` in the parameter
*defaults*. The default carries the graph edge; the `pl.Expr` type hint describes
what the body sees.

```python
import polars as pl
from schenberg.core.graph import Formula, Term, uses
from schenberg.domain.base import SchenbergDataFrameModel
from schenberg.market_data.requirements import MarketRequirements, requires
from schenberg.pricing.market import CURVES, VOL

class MyTrade(SchenbergDataFrameModel):
    spot: float
    strike: float
    payment_days: int

class MyOutput(SchenbergDataFrameModel):
    price: float

class CallRequirements(MarketRequirements[MyTrade]):
    rate: Term[float] = requires(CURVES.zero_rate())
    vol: Term[float] = requires(VOL.implied_vol())

g = Formula[MyTrade, CallRequirements, MyOutput]("my_call")
c, m = g.contract, g.market

@g.formula(symbol="T", latex=r"\frac{d}{252}")
def year_fraction(d: Term[int] = uses(c.payment_days)) -> pl.Expr:
    return d / 252.0

@g.formula(symbol="C")
def price(
    S: Term[float] = uses(c.spot),
    r: Term[float] = uses(m.rate),
    sigma: Term[float] = uses(m.vol),
    T: Term[float] = uses(year_fraction),
) -> pl.Expr:
    ...  # a Polars expression

g.returns()  # MyOutput fields are satisfied by like-named terms
```

`returns(schema=None)` publishes the primary `output` view (defaulting to the
`Output` type parameter); `view(name, schema)` adds secondary typed views. Every
field is satisfied *by name* — the contract, market or formula term of the same
name — so there is no column mapping. `g.bind(trades, market=...)` resolves the
environment and `g.plan(bound)` returns one lazy plan; as a `Computation`,
`g.compute(frame, market=..., view="output")` slots into a `Router`/`Structure`.

## 3. Market data is declared once, in the requirements schema

A `Formula`'s formulas never join. *What market data the instrument needs and
how to find each row* lives in one place — a `MarketRequirements` schema — and the
`MarketSnapshot` supplies *where it comes from* at compute time (the Reader
environment, injected late).

```python
class CallRequirements(MarketRequirements[MyTrade]):
    rate: Term[float] = requires(CURVES.zero_rate())
    vol: Term[float] = requires(VOL.implied_vol())
```

Each field name *is* the market column the instrument exposes; `requires(...)`
wraps a fluent read from the source registry (`schenberg.pricing.market`). A read
is **lookup-oriented**: it declares *which value to read for each row*. A keyed
join (`CURVES.zero_rate()`) is one implementation; interpolation
(`VOL.implied_vol()`, vol surfaces) is another — both compile to the same
`MarketDependency`. `.by(key=contract.column)` overrides a join key; it is
optional, because each read carries typed default key columns, so you write it
only when a contract names its columns unconventionally.

At compute time the engine attaches the market **before** compiling formulas, so
**lookup keys must already be columns** on the input frame. Anything that derives a
lookup key (normalizing wide rows into legs, computing a reference date) is a
*pre-step* — a plain transform or a `Workflow` stage — not a formula.

## 4. Open-graph composition

Graphs compose as open graphs. The three operations:

- **`merge`** — *parallel* composition: combine two graphs in the same
  environment, no automatic wiring. `a.merge(b)`.
- **`extend` / `compose_with`** — *same-environment* formula extension: add a
  formula block that reads the same inputs/market. The common case for layering
  closed-form Greeks onto a price.
- **`then`** — *port* composition: feed one graph's outputs into another's inputs.
  `payoff.then(discounting, bind={discounting.input.future_value: payoff.output.future_value})`.

Shared identical terms are reused by name; two different formulas under one name
are a hard conflict. `FormulaGraph.identity()` is the unit for `then`.

## 5. Router as ArrowChoice

A `Router` is not a list of filters — it is a contract-oriented **choice among
computations** that all satisfy the *same view contract*. Every branch is a
`FormulaGraph` (or nested router) producing the declared view, so the result is
total over the contract no matter which branch a row takes.

```python
from schenberg.core.router import Router

router = (
    Router.on(t.option_model, t.option_kind)
    .returns("price", OptionPrice)
    .exclusive()
)

@router.case(OptionModel.GENERALIZED, OptionKind.CALL)
def _generalized_call():
    return generalized_call_graph

@router.when(t.option_model == OptionModel.MERTON, t.option_kind == OptionKind.CALL)
def _merton_call():
    return merton_call_graph
```

- `case(...)` is sugar for equality predicates on the route terms; `when(...)`
  takes arbitrary predicates.
- The default mode is **`exclusive`**: a duplicate `case` key is rejected at
  registration, and `diagnose(frame)` reports per-branch match counts so you can
  check the cases truly partition the rows. `first_match()` opts into priority
  order.
- Registering a branch checks it provides the contract view with a compatible
  schema. After routing, the concatenated output is **normalized to the contract
  columns**, so a relaxed concat can never silently widen the result.

The implementation still filters per branch and `concat`s — but the semantics are
"choose among computations with one contract".

## 6. Structure: component pricing + exposure + Fold

A `FormulaGraph` prices **pure components** and must never know *position
direction* — no `side`, `pay_receive`, `ativo`/`passivo`, `long`/`short`, or
signed PV. A swap *is* its legs: each leg is priced purely, then weighted, then
aggregated. That composition is a `Structure`:

```
component pricing  →  exposure/weighting  →  fold
```

```python
import polars as pl
from schenberg.core.structure import Structure
from schenberg.core.fold import sum_

swap_structure = (
    Structure("swap", input=SwapLegInput)
    .components(swap_leg_router, view="pricing")           # pure LegPricing per leg
    .exposure(weighted_pv=pl.col("pv") * pl.col("leg_weight"))  # direction lives here
    .fold(
        by="swap_id",
        returns=SwapOutput,
        npv=sum_("weighted_pv"),
        ativo_pv=sum_("weighted_pv", where=L.leg_role == "ativo"),
        passivo_pv=sum_("weighted_pv", where=L.leg_role == "passivo"),
    )
)
```

- `components_frame(frame, market=...)` → the **pure** component prices (no sign).
- `stage(frame, market=...)` → component prices **plus** exposure (`weighted_pv`).
- `compute(frame, market=...)` → the folded output, one row per `swap_id`.
- `explain()` / `info()` / `to_mermaid()` describe the whole pipeline.

The pure leg `pv` is invariant to `leg_weight`; only the structure's exposure and
fold turn it into a signed contribution. `ativo_pv` / `passivo_pv` are *fold
classifications*, not pricing formulas.

## 7. Fold: monoidal aggregation

A `Fold` is the one place "group component rows by key and combine their values"
lives — no ad-hoc `group_by(...).agg(...)` scattered through pricers. Declare the
group keys, the output schema, and one aggregation per column with the `sum_`,
`first_`, `count_`, `lit_` helpers:

```python
from schenberg.core.fold import Fold, sum_, lit_, first_

forward_value_fold = (
    Fold("forward_value", input_schema=ForwardPricing)
    .by(F.instrument_id)
    .returns(
        InstrumentValue,
        instrument_type=lit_("FORWARD"),
        value=sum_(P.value),
        currency=first_(P.currency),
    )
)
valued = forward_value_fold.compute(component_rows)   # lazy
```

Aggregations are *monoidal* (associative reductions with an empty-group unit), and
inspectable — `explain()` renders `npv = sum(weighted_pv)` rather than an opaque
expression. The same `Fold` powers both structured instruments (via `Structure`)
and portfolio/book roll-ups.

## 7b. PositionView: measures over a position

Pricing answers *what is one unit worth?* and returns a **pure** `InstrumentValue`
(no `side`, no position). The position layer answers *how much do I hold, and what
is that worth in my book's terms?*. A `PositionView` has the same shape as a
`Formula`, only the boundary is wider: a **spine** frame (the `Position`) plus
**context sources** (the `InstrumentValue`, the `BookContract`, the `ReportingFx`)
joined *before* compilation, then pure row-local **measures** — `exposure`, `mtm`,
`reported_mtm` — declared exactly like pricing formulas.

```python
P, V, FX = position_value.position, position_value.value, position_value.fx

@position_value.measure(symbol="MTM")
def mtm(e=uses(exposure), val=uses(V.value)) -> pl.Expr:
    return e * val            # exposure carries the direction; pricing never does
```

Because the measures are terms in an internal `FormulaGraph`, the view gets
`explain()` / `info()` / `to_mermaid()` / `stage()` for free, and stays lazy.
`side` / `quantity` live on the `Position` and enter only here. Reporting currency
is a *measure* (`mtm / book_fx`), not a pricing concern.

**The view is generic over the pure per-instrument quantity it lifts.** Swap
`InstrumentValue` for any other pure pricing output and the same machinery applies:

| View | Joined quantity | Measures |
|------|-----------------|----------|
| `position_value` | `InstrumentValue` (`value`) | `exposure`, `mtm = exposure*value`, `reported_mtm = mtm/book_fx` |
| `position_pnl_explain` | `InstrumentPnlExplain` (`*_value_pnl`) | `<c>_mtm_pnl = exposure*<c>_value_pnl/book_fx`, `total = Σ` |
| `position_risk` | `InstrumentRisk` (the Greeks) | `position_<greek> = exposure * <greek>` |

So **PnL explain** is a derived measure, never the definition of a position; and
**risk factors** are just another pure per-instrument vector lifted by exposure —
the one primitive is `scaled(column, by="exposure")` (`mtm` *is*
`scaled("value")`; a position Greek *is* `scaled("delta")`). Reporting-currency
conversion of a currency-valued factor is the same `/ book_fx` step.

**Book/portfolio roll-up is a later layer**: a `Fold` over the view's output
(`book_value_rollup`), so a position is never confused with an aggregate.

## 8. Shock and MarketPath

A `Shock` is an **endomorphism** `MarketSnapshot → MarketSnapshot`: it returns a
new snapshot with sources transformed, never mutating the original, and shocks
**compose**. A `MarketPath` is a lens-lite that focuses one source/column to build
one:

```python
from schenberg.market_data.path import MarketPath
from schenberg.market_data.shocks import Shock

bump = MarketPath("curves").column("zero_rate").modify(lambda r: r + 1e-4)
stressed = market.apply(bump)                 # original market untouched
scenario = Shock.compose(bump, vol_bump)      # stresses chain associatively
```

Shocks preserve source schemas and expose `explain()` / `info()`. Repricing under
a stressed market is just `price_*(trades, market.apply(shock))`.

## 9. Workflow: shape-changing stages

A `FormulaGraph` is row-local (one column expression). When shapes change between
steps — joins, `group_by`, repricing under a bumped market — use a `Workflow`: a
DAG of stage functions returning LazyFrames, dependencies inferred from parameter
names. Nothing collects.

```python
from schenberg.core.pipeline import Workflow

workflow = Workflow("portfolio_pricing")

@workflow.stage
def normalized_trades(raw):
    return raw.unpivot(...)

@workflow.stage
def prices(normalized_trades):
    return price_options(normalized_trades, market)

env = workflow.run(raw=book)   # {"raw", "normalized_trades", "prices"}
```

## 10. Debugging and introspection

The *same* declaration powers execution and every report — they cannot drift.
Graphs, routers, structures, folds, shocks and workflows are all explainable:

- `graph.dependencies_of(term)` → the transitive inputs.
- `graph.required_inputs()` → the columns a caller must supply.
- `graph.formula_of(term)` / `graph.formulas()` → the `latex` math labels.
- `graph.info(view="price")` → inputs, market outputs, formula and view terms.
- `graph.explain(view="price")` → inputs, market reads, formula path and returns.
- `graph.to_mermaid(math_labels=True, show_kinds=True, view="price")` → a diagram
  classifying input / market / formula / output terms.
- `graph.stage(frame, market=..., view="price")` → materialize *every* intermediate
  as its own column; nulls propagate, so the first unexpectedly-null column is the
  root cause.
- `structure.explain()` / `.to_mermaid()` → input → component graph/router →
  exposure → fold → output; `structure.components_frame(...)` and `.stage(...)`
  expose the pure prices and the weighted contributions.
- `fold.explain()` / `router.explain()` / `router.diagnose(frame)` /
  `workflow.explain()` / `shock.explain()` round out the picture.

When a check wants to *accumulate* problems rather than raise on the first, a
`DiagnosticReport` collects `Diagnostic`s (`add`/`extend`), answers `has_errors`,
can `raise_if_errors()`, and renders `to_frame()`.

## Contracts at the boundary

Type hints help authors, but **Pandera remains the runtime contract boundary**.
`SchenbergDataFrameModel` (the base class for every boundary schema) runs
`@rule_for` contract rules before Pandera validates — filling derived date
columns like `index_fixing_date` from contract terms. Pricing functions are
annotated with `@pa.check_types(lazy=True)`; no manual `.resolve()` call is
needed. Inside the engine it's plain Polars expressions, so the hot path stays
free of per-node validation.

## Router vs data

> **Different curve *values* → data (a join key).
> Different *formula or set of sources* → a `Router`.**

A `Router` changes the *expression tree*. A join key changes the *numbers fed into
it*. If two instruments compute the same way and only read different curve points,
don't route — stack the curves in one table keyed by an identity column and let the
join pick the right rows. Even a convention that *looks* like branching ("IPCA reads
the index on Jan 1, CPI in April") is still data when it reduces to a different
*value* in a column — see the
[custom instrument example](examples/custom_instrument/index.md).

## Layers

```
domain/        Pandera boundary schemas + enums                              (no deps)
core/          Term, FormulaGraph, Router, Structure, Fold, MarketRead,
               Workflow, DiagnosticReport                                    (the engine)
market_data/   MarketSnapshot, sources, curve/vol specs, MarketPath, Shock
pricing/       instruments (swap, forward, option, ...) + portfolio
```

Dependencies point downward only: `pricing → market_data → core → domain`.
