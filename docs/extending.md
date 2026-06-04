# Extending the toolkit

The engine is the product. Adding an instrument, an index, or a payoff variant
is meant to be a small, local change — never an engine edit. This guide walks the
patterns, using the worked
[inflation-linked energy forward](../examples/custom_instrument/) as the running
example.

## Add a new payoff: write a graph

An instrument is "another graph". Declare it over an input schema, name the market
data as terms with `g.market`, wire formulas with `uses`, and publish a view:

```python
import polars as pl
from schenberg.core.graph import FormulaGraph, uses
from schenberg.market_data.curves import CurveSpec

g = FormulaGraph("my_instrument", input=MyTrade)
t = g.input
m = g.market(forward_price=CurveSpec("a_curve").value("price",
                                                      indexer=t.id_indexador,
                                                      tenor=t.payment_days))

@g.formula()
def payoff(fwd: pl.Expr = uses(m.forward_price), k: pl.Expr = uses(t.strike)) -> pl.Expr:
    return fwd - k

g.returns("pricing", value=payoff)
```

The full worked version is
[`examples/custom_instrument/graph.py`](../examples/custom_instrument/graph.py).
Reuse shared math from `schenberg.math.expressions`, and `extend()` /
`compose_with()` existing graphs to inherit their formulas rather than copy them.

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

3. Used as a join key in a `MarketRequirement` whose `on` binds
   `("reference_date", "ref_date")` (see `examples/custom_instrument/graph.py`).

Because the convention reduces to a *value* in a column, **one graph prices every
index** — no router, no branching in the formulas.

## Aggregate and expose

Finish with a thin public function that normalizes, runs the graph, and
aggregates to the level you report at — mirroring the built-in pricers:

```python
def price_my_instrument(legs, market):
    prepared = add_join_keys(legs)
    priced = g.compute(prepared, market=market, view="pricing")
    return priced.group_by("instrument_id").agg(price=pl.col("value").sum())
```

## Fixing / reference-date adaptation

Sometimes a convention only changes *which market row* is selected — e.g.
the PCA fixing factor must be read at the first day of the contract's tenor
year, not at the tenor date itself.  That is a join-key derivation, not a
router or graph change.

1. Derive the join-key column **before** the graph using the expression
   helpers in `schenberg.market_data.date_rules`:

   ```python
   from schenberg.market_data.date_rules import start_of_tenor_year
   from schenberg.market_data.fixings import FixingsSpec

   FIXINGS = FixingsSpec("fixings")

   prepared = legs.with_columns(
       start_of_tenor_year(tenor_col="tenor_date", output_col="pca_fixing_date")
   )
   ```

2. Declare it as a market term, naming the output by keyword:

   ```python
   m = graph.market(
       pca_factor=FIXINGS.value(indexer="id_indexador", date="pca_fixing_date"),
   )
   ```

The graph formula reads `m.pca_factor` like any other market term; it never knows
which calendar convention was applied.  Adding a new convention is a one-line
expression at the call site.

## Structured products

A structured product is a table of component legs.  Price each component
normally, then combine with `price_structures`:

```python
from schenberg.pricing.structured import price_structures
from schenberg.position.functions import with_prices

atomic_prices = pl.concat([forward_prices, swap_prices], how="diagonal_relaxed")
structure_prices = price_structures(structure_legs, atomic_prices)
all_prices = pl.concat([atomic_prices, structure_prices], how="diagonal_relaxed")
priced_positions = with_prices(positions, all_prices)
```

`structure_legs` is a `StructureLeg` frame with columns
`structure_id, leg_id, component_instrument_type, component_instrument_id,
quantity, side`.  The output of `price_structures` has the same shape as
any other `InstrumentPrice` frame (`instrument_type, instrument_id, price`)
with `instrument_type = "STRUCTURE"`.

## Checklist

- [ ] New math → a graph (`FormulaGraph`), reusing `extend()` / `compose_with()` / shared expressions.
- [ ] Per-row formula forks → a `Router`; pure value differences → a join key.
- [ ] Join keys not in the raw input → a transform/stage *before* the graph.
- [ ] Index/convention differences → a data registry, extended by one row.
- [ ] Fixing-date convention → `date_rules` expression, not a router case.
- [ ] Structured product → `price_structures`, not a new graph or router.
- [ ] Public function returns a lazy frame; `collect()` stays the caller's call.
- [ ] Boundary typed with a Pandera schema; internals stay plain Polars.
