# Concepts

Schenberg is a small engine for **composable, lazy pricing**. You describe
pricing as a graph of formulas; the engine compiles it into a single lazy Polars
expression and never collects until you ask. This page is the mental model.

## The four building blocks

| Block | What it is | When you reach for it |
|-------|------------|-----------------------|
| **`FormulaGraph`** | A DAG of row-local formulas that compiles to one nested `pl.Expr`. | Math where the shape doesn't change: discounting, payoffs, factors. |
| **`Router`** | Splits rows by predicate and sends each subset to a different pricer. | The *formula* differs per row (instrument family, option kind, ...). |
| **`MarketSnapshot`** | Named market sources + declarative market reads. | Pulling curves/fixings/FX onto the trade rows by key. |
| **`Workflow`** | A DAG of *stages* returning LazyFrames. | Steps that change shape: joins, group-bys, repricing under a bumped market. |

Everything stays lazy. `collect()` is yours to call, once, at the edge.

## FormulaGraph: formulas as a graph

A formula is a function whose **parameter names are its dependencies**. You never
wire edges by hand — the engine reads them from the signature.

```python
from schenberg.core.graph import FormulaGraph
import polars as pl

g = FormulaGraph("demo")

@g.formula()
def year_fraction(payment_days: pl.Expr) -> pl.Expr:
    return payment_days / 252.0

@g.formula()
def discount_factor(zero_rate: pl.Expr, year_fraction: pl.Expr) -> pl.Expr:
    return (-zero_rate * year_fraction).exp()
```

`discount_factor` depends on `zero_rate` (an input column) and `year_fraction`
(another formula). You name the output columns a graph exposes with `returns`
(each named set is a *view*), then `g.compute(frame, view="...")` compiles the
requested view into one `with_columns` call. Intermediates are shared across
outputs via a compile cache.

```python
g.returns("pricing", discount_factor="discount_factor")
priced = g.compute(frame, market=market, view="pricing")
```

Useful introspection, all derived from the graph itself (so it can't drift):

- `g.dependencies_of("discount_factor")` → the transitive inputs.
- `g.required_inputs()` → the columns a caller must supply.
- `g.to_mermaid()` → a diagram of the DAG.
- `g.stage(frame, ...)` → materialize *every* intermediate as its own column for
  debugging; nulls propagate, so the first unexpectedly-null column is the root
  cause.

## MarketSnapshot: declarative market data

A graph declares *what* market data it needs with `for_market`; the snapshot
supplies *where* it comes from. Each keyword is the **output column** the read
writes onto the frame:

```python
from schenberg.market_data.curves import CurveSpec

CURVES = CurveSpec("curves")

g.for_market(
    rate=CURVES.value("zero_rate", indexer="id_indexador", tenor="payment_days"),
)
```

`CURVES.value(...)` returns a `MarketRequirement` whose output column defaults to
the value column's name; `for_market` then renames it to the keyword (`rate`).
Multi-output joins (one read that writes several columns) can't be renamed by
keyword — attach them through the lower-level `g.uses_market(req1, req2)` escape
hatch. Every plain keyed spec (`CurveSpec`, `FxRatesSpec`, `FixingsSpec`) is a thin
wrapper over the shared `JoinSpec` join builder.

At pricing time the engine attaches the market **before** compiling formulas, so
**join keys must already be columns** on the input frame. Anything that derives
a join key (normalizing wide rows into legs, computing a reference date) is a
*pre-step* — a plain transform or a `Workflow` stage — not a formula.

## Router vs data

This is the decision people get wrong most often.

> **Different curve *values* → data (a join key).
> Different *formula or set of sources* → a `Router`.**

The Router changes the *expression tree*. A join key changes the *numbers fed
into it*. If two instruments compute the same way and only read different curve
points, don't route — stack the curves in one table keyed by an identity column
and let the join pick the right rows per instrument. Even a convention that
*looks* like branching (e.g. "IPCA reads the index on Jan 1, CPI in April") is
still data when it reduces to a different *value* in a column — see the
[custom instrument example](../examples/custom_instrument/README.md).

Reach for a `Router` when the math genuinely forks. Build it with `Router.on`
over the route columns, then register cases with the `case` decorator (equality
on the route columns) or `when` (arbitrary predicates):

```python
from schenberg.core.router import Router
from schenberg.core.columns import cols

R = cols(MySchema)
router = Router.on(R.instrument_family).default(generic_graph)

@router.case("ENERGY")
def energy_graph():
    return FormulaGraph.compose("energy", base).uses_market(...)

@router.when(R.instrument_family == "EXOTIC", R.tenor_days > 365)
def long_dated_exotic():
    return exotic_graph
```

Unmatched rows fall to `.default(...)`. The fallback is **permissive by
design** — the boundary Pandera contracts already guard shape and types, so the
router stays about *dispatch*, not validation.

## Contracts at the boundary

Pandera schemas (`schenberg/domain`) type the public edges — inputs and outputs
of the pricing functions — and nothing internal. Inside the engine it's plain
Polars expressions. This keeps the hot path free of per-node validation while
still giving callers a typed contract.

## Layers

```
domain/        Pandera boundary schemas + enums                  (no deps)
core/          FormulaGraph, Router, MarketRequirement, Workflow (the engine)
market_data/   MarketSnapshot, sources, curve/vol specs, shocks
pricing/       instruments (swap, forward, option, ...) + portfolio
```

Dependencies point downward only: `pricing → market_data → core → domain`.

## Option market data and volatility surfaces

Market data belongs in market specs, not in pricing-facade enrichment. Curves,
fixings, dividends, carry curves and implied-volatility surfaces are all declared
by the graph through `for_market(...)`; pricing functions orchestrate contracts
and select public columns.

A volatility surface is interpolated over `(id_indexador, tenor_days, strike)`,
not a simple left join. The caller supplies the surface as a `MarketSource`
through `VolSurfaces.source()`, and the option graph declares the implied-vol
column:

```python
from schenberg.core.columns import cols
from schenberg.domain.schemas.option import OptionTrade
from schenberg.market_data.volatility import VolSurfaceSpec

OPT = cols(OptionTrade)
VOL = VolSurfaceSpec("vol_surface")

graph.for_market(
    vol=VOL.implied_vol(
        indexer=OPT.id_indexador,
        tenor=OPT.payment_days,
        strike=OPT.strike,
    )
)
```

`price_options(...)` is price-only and returns the public `OptionPrice` contract
(view `price`). `price_options_with_greeks(...)` is the separate public path for
sensitivities. Closed-form Greeks are a graph composition on top of the option
price graph (view `risk`); numeric and autodiff Greeks consume an explicit
priced-state contract (view `state`) containing price inputs and derived BSM
terms.

## Graph documentation and debugging

Formula metadata is documentation/introspection only; it does not parse Polars
expressions and does not change execution. Formulas may carry a `symbol` and a
`latex` math representation, then the graph can explain itself:

- `graph.formula_of("d1")` returns the math label for one formula.
- `graph.formulas()` returns labels for every formula.
- `graph.info(view="state")` summarizes required inputs, market inputs/outputs,
  formula nodes and the selected view.
- `graph.explain(view="state")` prints the dependency path in topological order.
- `graph.to_mermaid(math_labels=True, show_kinds=True, view="state")` emits a
  Mermaid diagram with formula labels and simple node classes.
- `graph.view_dtypes("state")` is the declared dtype contract for a view.

Use `graph.stage(...)` when you need materialized intermediate columns for null
or data-quality debugging; it remains the low-level LazyFrame inspection tool.

## Migration from the early API

This is an early library and the public vocabulary was simplified in a breaking
refactor. The mapping:

```text
ExprGraph        -> FormulaGraph
.node            -> .formula
formula=         -> latex=
.with_outputs    -> .returns
output_profile=  -> view=
.compute_for     -> .compute
.with_market     -> .for_market   (positional reqs -> .uses_market)
Pipe             -> Workflow
Router.by        -> Router.on
Router.register  -> Router.case / Router.when
```
