# Extending Schenberg

Keep extensions small, local, inspectable, and lazy.

## Add a new pricer

1. Declare `MarketRole`s for the market columns you need.
2. Build an input schema with `With[role]` mixins.
3. Build a `FormulaGraph` over those input columns.
4. Register formulas with `@g.formula(...)`.
5. Resolve market data with `bind` and run `graph.plan(...)`.

```python
import polars as pl
from schenberg import FormulaGraph, MarketSnapshot, With, bind, exp, market_role
from schenberg.domain.base import SchenbergDataFrameModel

MyRate = (
    market_role("my_rate")
    .read("my_curves", "rate")
    .by(indexer="id_indexador", payment_days="tenor_days")
)

class MyInput(With[MyRate], SchenbergDataFrameModel):
    instrument_id: str
    indexer: str
    payment_days: int
    strike: float

g = FormulaGraph("my_pricer", input=MyInput)

@g.formula(symbol="T")
def year_fraction(c):
    return c.payment_days / 252.0

@g.formula(symbol="DF")
def discount_factor(c, year_fraction):
    return exp(-c.my_rate * year_fraction)

@g.formula(symbol="FV")
def future_value(c):
    return c.my_rate - c.strike

@g.formula(symbol="PV")
def present_value(future_value, discount_factor):
    return future_value * discount_factor

g.returns("output", instrument_id="instrument_id", value="present_value")

def price_my_instrument(trades: pl.LazyFrame, market: MarketSnapshot) -> pl.LazyFrame:
    enriched = bind(trades, market, MyInput)
    return g.plan(enriched, view="output")
```

The decorator preserves the current symbolic IR: LaTeX, Mermaid, `explain`,
`stage`, required-input analysis, lazy Polars execution, and future derivative
support continue to work. Use `g.let(...)` only when you need the lower-level
primitive directly.

## Reuse a graph with another market source

Reuse the same formula by defining a different input schema whose market roles
publish the same semantic columns. Keep this wiring in your application or in an
example notebook; do not create a central module of example pricers.

## Add a market fixing

A `Fixing` derives a date join key from contract columns. The derived key is used
by `bind` before the graph runs:

```python
from schenberg import Fixing, market_role
from schenberg.market_data.date_rules import previous_business_days

PtaxFixing = (
    market_role("ptax")
    .read("ptax_fixings", "fixing_value")
    .by(currency_pair="currency_pair")
    .fixing("ptax_fixing_date", Fixing.rule(previous_business_days("tenor", n=5)))
)
```

`previous_business_days` uses a simplified Monday-Friday calendar. For production
calendars, provide a precomputed fixing date or a more specific date-rule helper.

## Add a position measure

Custom measures are small callables that register one term on a `PositionView`:

```python
from schenberg.position.view import Measure, PositionView
from schenberg.core.expr import Expr


def my_measure(*, name: str = "my_col") -> Measure:
    def register(view: PositionView) -> Expr:
        return view.let(name, view.col("exposure") * view.col("value") * 0.5)
    return Measure(register)
```

Position measures may use `side` and `quantity`; pure pricing graphs should not.

## HTML examples

Export notebooks directly with Marimo:

```bash
uv run marimo export html docs/examples/01_forward_pricer.py -o docs/examples/01_forward_pricer.html
uv run marimo export html docs/examples/02_forward_positions.py -o docs/examples/02_forward_positions.html
uv run marimo export html docs/examples/03_usdbrl_df_fixing.py -o docs/examples/03_usdbrl_df_fixing.html
```
