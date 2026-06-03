# Concepts

Schenberg is a small engine for **composable, lazy pricing**. You describe
pricing as a graph of formulas; the engine compiles it into a single lazy Polars
expression and never collects until you ask. This page is the mental model.

## The four building blocks

| Block | What it is | When you reach for it |
|-------|------------|-----------------------|
| **`ExprGraph`** | A DAG of row-local formulas that compiles to one nested `pl.Expr`. | Math where the shape doesn't change: discounting, payoffs, factors. |
| **`Router`** | Splits rows by predicate and sends each subset to a different pricer. | The *formula* differs per row (instrument family, option kind, ...). |
| **`MarketSnapshot`** | Named market sources + declarative `require(...)` joins. | Pulling curves/fixings/FX onto the trade rows by key. |
| **`Pipe`** | A DAG of *stages* returning LazyFrames. | Steps that change shape: joins, group-bys, repricing under a bumped market. |

Everything stays lazy. `collect()` is yours to call, once, at the edge.

## ExprGraph: formulas as a graph

A node is a function whose **parameter names are its dependencies**. You never
wire edges by hand — the engine reads them from the signature.

```python
from schenberg.core.graph import ExprGraph
import polars as pl

g = ExprGraph("demo")

@g.node()
def year_fraction(payment_days: pl.Expr) -> pl.Expr:
    return payment_days / 252.0

@g.node()
def discount_factor(zero_rate: pl.Expr, year_fraction: pl.Expr) -> pl.Expr:
    return (-zero_rate * year_fraction).exp()
```

`discount_factor` depends on `zero_rate` (an input column) and `year_fraction`
(another node). `g.compute_for(lf, ...)` compiles the requested outputs into one
`with_columns` call. Intermediates are shared across outputs via a compile cache.

Useful introspection, all derived from the graph itself (so it can't drift):

- `g.dependencies_of("discount_factor")` → the transitive inputs.
- `g.required_inputs()` → the columns a caller must supply.
- `g.to_mermaid()` → a diagram of the DAG.
- `g.stage(lf, ...)` → materialize *every* intermediate as its own column for
  debugging; nulls propagate, so the first unexpectedly-null column is the root
  cause.

## MarketSnapshot: declarative market data

A graph declares *what* market data it needs; the snapshot supplies *where* it
comes from. `require(...)` is a left join described by key bindings.

```python
g.with_market(
    require("di_curve", ("id_indexador", "id_indexador"), ("payment_days", "tenor_days"),
            outputs={"zero_rate": "zero_rate"}),
)
```

At pricing time the engine attaches the market **before** compiling formulas, so
**join keys must already be columns** on the input frame. Anything that derives
a join key (normalizing wide rows into legs, computing a reference date) is a
*pre-step* — a plain transform or a `Pipe` stage — not a graph node.

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

Reach for a `Router` when the math genuinely forks:

```python
from schenberg.core.router import Router
from schenberg.core.columns import cols

R = cols(MySchema)
router = Router.by(R.instrument_family).default(generic_graph)

@router.register(R.instrument_family == "ENERGY")
def energy_graph():
    return ExprGraph.compose("energy", base).with_market(...)
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
domain/        Pandera boundary schemas + enums            (no deps)
core/          ExprGraph, Router, MarketRequirement, Pipe  (the engine)
market_data/   MarketSnapshot, sources, curve specs, shocks
pricing/       instruments (swap, forward, ...) + portfolio
```

Dependencies point downward only: `pricing → market_data → core → domain`.

## Option market data and volatility surfaces

Market data belongs in market specs, not in pricing-facade enrichment. Curves,
fixings, dividends, carry curves and implied-volatility surfaces are all declared
by the graph through `with_market(...)`; pricing functions orchestrate contracts
and select public columns.

A volatility surface is a custom attachable market dependency because the lookup
is interpolation over `(id_indexador, tenor_days, strike)`, not a simple left
join. The caller supplies the surface as a `MarketSource` through
`VolSurfaces.source()`, and the option graph declares the implied-vol column:

```python
from schenberg.core.columns import cols
from schenberg.domain.schemas.option import OptionTrade
from schenberg.market_data.volatility import VolSurfaceSpec

OPT = cols(OptionTrade)
VOL = VolSurfaceSpec("vol_surface")

graph.with_market(
    VOL.implied_vol(
        indexer_col=OPT.id_indexador.name,
        tenor_col=OPT.payment_days.name,
        strike_col=OPT.strike.name,
        output="vol",
    )
)
```

`price_options(...)` is price-only and returns the public `OptionPrice` contract.
`price_options_with_greeks(...)` is the separate public path for sensitivities.
Closed-form Greeks are a graph composition on top of the option price graph;
numeric and autodiff Greeks consume an explicit priced-state contract containing
price inputs and derived BSM terms.

## Graph documentation and debugging

Formula metadata is documentation/introspection only; it does not parse Polars
expressions and does not change execution. Nodes may carry a `symbol` and
`formula`, then the graph can explain itself:

- `graph.formula_of("d1")` returns the math label for one node.
- `graph.formulas()` returns formulas for formula nodes.
- `graph.info(output_profile="priced_state")` summarizes required inputs,
  market inputs/outputs, formula nodes and selected outputs.
- `graph.explain(output_profile="priced_state")` prints the dependency path in
  topological order.
- `graph.to_mermaid(math_labels=True, show_kinds=True)` emits a Mermaid diagram
  with formula labels and simple node classes.

Use `graph.stage(...)` when you need materialized intermediate columns for null
or data-quality debugging; it remains the low-level LazyFrame inspection tool.
