# Extending the toolkit

The engine is the product. Adding an instrument, an index, or a payoff variant
is meant to be a small, local change — never an engine edit. This guide walks the
patterns, using the worked
[inflation-linked energy forward](../examples/custom_instrument/) as the running
example.

## Add a new payoff: write a graph

An instrument is a typed `Formula[Contract, Requirements, Output]`. Declare a
`MarketRequirements` schema for the market data (one `requires(...)` field per
market column), wire formulas with `uses` over `g.input` / `g.market`, and
publish the output:

```python
import polars as pl
from schenberg.core.graph import Formula, Term, uses
from schenberg.market_data.requirements import Key, Keyed, MarketRequirements, requires

class MyRequirements(MarketRequirements[MyTrade]):
    # a built-in registry read, or a hand-rolled Keyed read for a custom table:
    forward_price: Term[float] = requires(
        Keyed("a_curve", "price", (Key("tenor", quote_col="tenor_days", default="payment_days"),))
    )

g = Formula[MyTrade, MyRequirements, MyPricing]("my_instrument")
c, m = g.input, g.market

@g.formula(name="value")
def payoff(fwd: Term[float] = uses(m.forward_price), k: Term[float] = uses(c.strike)) -> pl.Expr:
    return fwd - k

g.returns()  # MyPricing fields are satisfied by like-named terms
```

The full worked version is
[`examples/custom_instrument/graph.py`](../examples/custom_instrument/graph.py).
Reuse shared math from `schenberg.math.expressions` and the shared `Term`
builders in `schenberg.pricing.discounting` rather than copying formulas.

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

## Keep pricing graphs pure — direction lives in a `Structure`

A `FormulaGraph` prices **pure components**. It must never compute signed PV from
`side` / `pay_receive` / `ativo` / `passivo` / `long` / `short`. A swap leg graph
computes `pv = cashflow_amount * discount_factor` — no sign. The position
direction (`leg_weight`) is applied one layer up, in a `Structure`'s exposure, and
the ativo/passivo split is a `Fold` classification. See
[`concepts.md`](concepts.md#6-structure-component-pricing--exposure--fold).

## Aggregate and expose with a `Fold`

Finish with a thin public function that normalizes, runs the graph/structure, and
aggregates with a `Fold` (not an ad-hoc `group_by(...).agg(...)`) to the level you
report at — mirroring the built-in pricers:

```python
from schenberg.core.fold import Fold, sum_, lit_

my_fold = (
    Fold("my_instrument", input_schema=MyPricing)
    .by("instrument_id")
    .returns(InstrumentPrice, instrument_type=lit_("MY_TYPE"), price=sum_("value"))
)

def price_my_instrument(legs, market):
    prepared = add_lookup_keys(legs)
    priced = g.compute(prepared, market=market, view="pricing")
    return my_fold.compute(priced)
```

For a structured instrument (component pricing + exposure + fold), assemble a
`Structure` and call `structure.compute(legs, market=market)` — see the swap
structure in `schenberg/pricing/instruments/swap/structure.py`.

## Fixing / reference-date adaptation

Sometimes a convention only changes *which market row* is selected — e.g.
the PCA fixing factor must be read at the first day of the contract's tenor
year, not at the tenor date itself.  That is a join-key derivation, not a
router or graph change.

1. Derive the join-key column **before** the graph using the expression
   helpers in `schenberg.market_data.date_rules`:

   ```python
   from schenberg.market_data.date_rules import start_of_tenor_year

   prepared = legs.with_columns(
       start_of_tenor_year(tenor_col="tenor_date", output_col="pca_fixing_date")
   )
   ```

2. Point a requirements field at the derived key with `.by(...)`:

   ```python
   from schenberg.market_data.requirements import MarketRequirements, contract, requires
   from schenberg.pricing.market import FIXINGS

   class MyRequirements(MarketRequirements[MyTrade]):
       pca_factor: Term[float] = requires(FIXINGS.value().by(date=contract.pca_fixing_date))
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

## Contract rules and derived contractual coordinates

Contractual coordinates (`index_fixing_date`, `currency_fixing_date`,
`projection_date`, …) differ from market values: they are determined by the
*contract terms*, not by the market.  Declare them with `@rule_for` on a
`SchenbergDataFrameModel` subclass or mixin; `validate()` resolves them
automatically before Pandera checks run.

```python
from schenberg.domain.base import SchenbergDataFrameModel
from schenberg.domain.rules import rule_for
from schenberg.market_data import date_rules as dates

class IndexerFixingMixin(SchenbergDataFrameModel):
    indexer: IndexerEnum
    index_fixing_date: date | None = None

    @rule_for("index_fixing_date", selector="indexer", value=IndexerEnum.CPI)
    def _cpi(cls):
        return dates.add_days("tenor", 5)

    @rule_for("index_fixing_date", selector="indexer", default=True)
    def _default(cls):
        return dates.same_day("tenor")
```

**Key rules:**

* Use `SchenbergDataFrameModel` as the base class for every boundary schema.
* Mixins can declare rules; multiple mixins compose cleanly — each output/selector
  pair is independent.
* User-provided non-null values are always preserved (coalesce semantics).
* Null or missing values are filled lazily — no `.collect()` is called.
* `pa.check_types(lazy=True)` on pricing functions is the only annotation needed;
  no extra decorator and no manual `resolve()` call.
* Formula graphs never compute fixing dates — they only read market values.
* `MarketSnapshot` provides market *values* at contractual *coordinates*; it does
  not derive the coordinates themselves.
* Child schemas can override a parent's rule case for the same
  `(output, selector, value)` triple by redeclaring it.

## Checklist

- [ ] New math → a graph (`FormulaGraph`), reusing `extend()` / `compose_with()` / shared expressions.
- [ ] Position direction (`side`/`pay_receive`/weight) → a `Structure`'s exposure, **never** the pricing graph.
- [ ] Per-row formula forks → a `Router`; pure value differences → a lookup key.
- [ ] "Group by id and combine values" → a `Fold` (or a `Structure`'s fold), not ad-hoc `group_by`.
- [ ] Lookup keys not in the raw input → a transform/stage *before* the graph.
- [ ] Index/convention differences → a data registry, extended by one row.
- [ ] Fixing-date convention → `date_rules` expression, not a router case.
- [ ] Instrument made of weighted components → a `Structure`; cross-instrument book → `price_structures`.
- [ ] Scenario/stress → a `Shock` (built via `MarketPath`), applied with `market.apply(shock)`.
- [ ] Public function returns a lazy frame; `collect()` stays the caller's call.
- [ ] Boundary typed with a Pandera schema; internals stay plain Polars.
