# Extending the toolkit

The engine is the product. Adding an instrument, an index, or a payoff variant
is meant to be a small, local change — never an engine edit. This guide walks the
patterns, using the worked
[inflation-linked energy forward](../examples/custom_instrument/) as the running
example.

## Add a new payoff: write a graph

An instrument is "another graph". Define nodes whose parameter names are their
dependencies, declare the market data the graph needs, and name an output
profile:

```python
g = ExprGraph("my_instrument")

@g.node()
def payoff(forward_price: pl.Expr, strike: pl.Expr) -> pl.Expr:
    return forward_price - strike

g.with_outputs("pricing", value="payoff")
g.with_market(require("a_curve", ("key", "key"), outputs={"price": "forward_price"}))
```

Reuse shared math from `schenberg.math.expressions`, and `compose()` existing
graphs to inherit their nodes rather than copy them.

## Add a new variant of the same payoff: route it

When the math forks per row, register cases on a `Router` keyed by a column.
Each case composes the shared graph and configures its own market. Unmatched
rows hit `.default(...)`. See [`concepts.md`](concepts.md#router-vs-data) for
when routing is the right tool versus just data.

## Add a new market dimension: it's a join key, computed up front

The most common subtlety: a value the graph needs to *join on* but that isn't in
the raw input. Market attach happens **before** formula compilation, so derive
join keys in a **pre-step**, not a node.

In the example, the inflation factor must be read at an index-specific calendar
date. That `reference_date` is:

1. Defined as data — a registry of conventions (`conventions.py`):

   ```python
   CONVENTIONS = (
       InflationConvention(id_indexador=20, name="IPCA", reference_month=1),
       InflationConvention(id_indexador=10, name="CPI",  reference_month=4),
   )
   ```

   Onboarding a new index is one row here.

2. Built as a single lazy column via `reference_date_expr()` and attached by a
   transform (`add_reference_date`) that runs before the graph.

3. Used as a join key: `require("inflation_curve", ("id_indexador", "id_indexador"),
   ("reference_date", "ref_date"), ...)`.

Because the convention reduces to a *value* in a column, **one graph prices every
index** — no router, no branching in the formulas.

## Aggregate and expose

Finish with a thin public function that normalizes, runs the graph, and
aggregates to the level you report at — mirroring the built-in pricers:

```python
def price_my_instrument(legs, market):
    prepared = add_join_keys(legs)
    priced = g.compute_for(prepared, market=market, output_profile="pricing")
    return priced.group_by("instrument_id").agg(price=pl.col("value").sum())
```

## Checklist

- [ ] New math → a graph (`ExprGraph`), reusing `compose()` / shared expressions.
- [ ] Per-row formula forks → a `Router`; pure value differences → a join key.
- [ ] Join keys not in the raw input → a transform/stage *before* the graph.
- [ ] Index/convention differences → a data registry, extended by one row.
- [ ] Public function returns a lazy frame; `collect()` stays the caller's call.
- [ ] Boundary typed with a Pandera schema; internals stay plain Polars.
