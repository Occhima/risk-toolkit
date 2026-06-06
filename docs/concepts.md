# Concepts

Schenberg is a small lazy pricing DSL over Polars expressions.

## Terms and formula graphs

A `FormulaGraph` is a directed acyclic graph of named terms:

- **Input terms** — columns supplied by the trade frame (after market binding).
- **Formula terms** — `Expr` nodes over earlier terms, compiled to
  `pl.with_columns` at plan time.
- **Views** — named projections of the graph onto an output schema.

`plan(frame, view="output")` returns a `pl.LazyFrame`. It validates that
required input columns are present, adds view expressions as `with_columns`,
and never collects. The graph never reads market data — market columns arrive
pre-resolved as ordinary input columns.

`stage(frame, view="output")` materialises every intermediate term as its own
column, for debugging.

```python
from schenberg.core.graph import FormulaGraph
from schenberg.core.expr import exp, var

g = FormulaGraph("forward_pricing")
c = g.input
T  = g.let("year_fraction",   c.payment_days / 252.0,         symbol="T")
DF = g.let("discount_factor", exp(-c.risk_free_rate * T),      symbol="DF")
FV = g.let("future_value",    c.forward_rate - c.strike,       symbol="FV")
PV = g.let("present_value",   FV * DF,                         symbol="PV")
g.let("value", PV, symbol="V")
g.returns("output", future_value="future_value", present_value="present_value", value="value")
```

The same graph is inspectable without pricing anything:

```python
g.explain(view="output")          # human-readable formula tree
g.to_mermaid()                    # Mermaid flowchart
g.required_inputs("output")       # {'forward_rate', 'payment_days', ...}
g.dependencies_of("present_value") # which terms does present_value depend on?
```

## Market data

A `MarketSnapshot` is the environment for market reads. A `MarketSource` can
carry `unique_by=(...)` to validate quote-key uniqueness during snapshot
construction — an explicit market-data boundary, never inside a pricing graph.

### Market roles

A `MarketRole` declares *one* market column a schema needs: which source to
read, which quote value, how to join (exact keys + optional date fixing), and
the column name it publishes into the enriched input frame.

```python
from schenberg.market_data.roles import market_role, With, bind

ForwardRate = (
    market_role("forward_rate")
    .read("curves", "forward_rate")
    .by(indexer="id_indexador", payment_days="tenor_days")
)

class MyInput(With[ForwardRate]):
    instrument_id: str
    indexer: str
    payment_days: int
    strike: float
    # forward_rate: float  ← added by bind()
```

`bind(trades, snapshot, MyInput)` discovers the `With[...]` mixins on
`MyInput` via `roles_of()`, runs each role's join against the snapshot, then
projects and validates against the schema. The pricing graph never sees the
snapshot.

## Pricing boundary

Pure pricing functions return own-currency values only. They do not read book
columns, position side, quantity, legal entity, reporting currency, or reported
MTM.

The public pricers:

```python
from schenberg.pricing.api import price_forward, price_energy_forward
```

Generic and energy forwards share the same formula graph — the specialisation
is entirely in which market roles they declare.

## Position boundary

The position layer consumes already-valued instruments:

- `InstrumentValue.value` is in `InstrumentValue.currency`.
- `Position.side * Position.quantity` creates exposure.
- `BookContract.reporting_currency` and `ReportingFx.book_fx` convert reported
  MTM (`reported_mtm = mtm / book_fx`).

```python
from schenberg.position import position_value, position_risk, book_value_rollup

# Lift one InstrumentValue onto a Position + Book + FX context
pv = position_value(positions, value=values, book=book, fx=fx)

# Lift InstrumentRisk (per-unit Greeks) onto positions
pr = position_risk(positions, risk=risk)

# Roll up to book level
rollup = book_value_rollup.compute(pv)
```

Aggregation (`Fold`) is a *later* layer than the position view and is never
part of it.

## Expression IR

All formula terms are `Expr` nodes from `schenberg.core.expr`. The same tree
is interpreted in three ways:

- **Polars** — `compile_polars(expr)` → `pl.Expr` (lazy execution).
- **Numeric** — `compile_numeric(expr, bindings)` → `float` (point evaluation).
- **LaTeX** — `to_latex(expr)` → `str` (human-readable math).

An analytic derivative is available via JAX when installed:
`grad(expr, "forward_rate")` produces the partial derivative as another `Expr`.

## Shocks and scenarios

A `Shock` is a pure `MarketSnapshot → MarketSnapshot` endomorphism. It never
mutates the original. `Shock.compose(*shocks)` chains them associatively:

```python
from schenberg.market_data.shocks import curve_parallel_shift
from schenberg.market_data.path import MarketPath

bump = curve_parallel_shift(source="curves", column="risk_free_rate", shift=0.01)
stressed = market.apply(bump)          # new snapshot, original untouched
```
