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
from schenberg.core.graph import FormulaGraph, uses
from schenberg.domain.schemas.option import OptionTrade, OptionPrice
from schenberg.market_data.curves import CurveSpec
from schenberg.market_data.volatility import VolSurfaceSpec

g = FormulaGraph("generalized_call", input=OptionTrade)
t = g.input

m = g.market(
    rate=CurveSpec("curves").value("zero_rate", indexer=t.id_indexador, tenor=t.payment_days),
    vol=VolSurfaceSpec("vol_surface").implied_vol(
        indexer=t.id_indexador, tenor=t.payment_days, strike=t.strike
    ),
)

@g.formula(symbol="T", latex=r"\frac{d}{252}")
def year_fraction(d: pl.Expr = uses(t.payment_days)) -> pl.Expr:
    return d / 252.0

@g.formula(symbol="C")
def call_price(
    S: pl.Expr = uses(t.spot),
    r: pl.Expr = uses(m.rate),
    sigma: pl.Expr = uses(m.vol),
    T: pl.Expr = uses(year_fraction),
) -> pl.Expr:
    ...  # a Polars expression

g.returns("price", OptionPrice, option_id=t.option_id, instrument_type=t.instrument_type,
          price=call_price)
```

`returns(view, schema, **mapping)` declares a typed result *view*: each schema
field must be satisfied by a term (pass it explicitly, or let an identically
named term fill it); extra columns are rejected. `g.compute(frame, market=...,
view="price")` then compiles the view into one lazy `with_columns`. A bare `Term`
default (`d=t.payment_days`) is accepted as shorthand, but `uses(...)` is the
canonical style because it keeps type hints clean.

## 3. Market data is the Reader environment — and it is lookup-oriented

Market reads are declared *inside the graph* as terms with `g.market(...)`; the
`MarketSnapshot` supplies *where they come from* at compute time — the Reader
environment, injected late.

```python
m = g.market(
    rate=CurveSpec("curves").value("zero_rate", indexer=t.id_indexador, tenor=t.payment_days),
)
```

A market read is **lookup-oriented**: it declares *which value to read for each
row*. A keyed join is only one implementation of that lookup; interpolation
(`InterpolatedSpec`, vol surfaces), fixings, and scenario reads are other
implementations of the same `MarketRead`/`MarketDependency` idea. Don't think of
market data as "join-oriented" — joins are one strategy among several.

`CurveSpec(...).value(...)` with no `output` returns a delayed **`MarketRead`** —
it knows *what* to read but not yet *where* to write it. `g.market(rate=...)`
names the output column from the keyword. At compute time the engine attaches the
market **before** compiling formulas, so **lookup keys must already be columns** on
the input frame. Anything that derives a lookup key (normalizing wide rows into
legs, computing a reference date) is a *pre-step* — a plain transform or a
`Workflow` stage — not a formula.

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
from schenberg.core.fold import Fold, sum_, lit_

forward_price_fold = (
    Fold("forward_price", input_schema=ForwardPricing)
    .by(F.instrument_id)
    .returns(InstrumentPrice, instrument_type=lit_("FORWARD"), price=sum_(P.value))
)
priced = forward_price_fold.compute(component_rows)   # lazy
```

Aggregations are *monoidal* (associative reductions with an empty-group unit), and
inspectable — `explain()` renders `npv = sum(weighted_pv)` rather than an opaque
expression. The same `Fold` powers both structured instruments (via `Structure`)
and portfolio/book roll-ups.

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
Pandera schemas (`schenberg/domain`) type the public edges — inputs and outputs of
the pricing functions — and nothing internal. Inside the engine it's plain Polars
expressions, so the hot path stays free of per-node validation.

## Router vs data

> **Different curve *values* → data (a join key).
> Different *formula or set of sources* → a `Router`.**

A `Router` changes the *expression tree*. A join key changes the *numbers fed into
it*. If two instruments compute the same way and only read different curve points,
don't route — stack the curves in one table keyed by an identity column and let the
join pick the right rows. Even a convention that *looks* like branching ("IPCA reads
the index on Jan 1, CPI in April") is still data when it reduces to a different
*value* in a column — see the
[custom instrument example](../examples/custom_instrument/README.md).

## Layers

```
domain/        Pandera boundary schemas + enums                              (no deps)
core/          Term, FormulaGraph, Router, Structure, Fold, MarketRead,
               Workflow, DiagnosticReport                                    (the engine)
market_data/   MarketSnapshot, sources, curve/vol specs, MarketPath, Shock
pricing/       instruments (swap, forward, option, ...) + portfolio
```

Dependencies point downward only: `pricing → market_data → core → domain`.
