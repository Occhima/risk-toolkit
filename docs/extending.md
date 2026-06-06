# Extending Schenberg

Keep extensions small and lazy.

## Add a new pricer

1. Declare `MarketRole`s for the market columns you need.
2. Build an input schema with `With[role]` mixins for each one.
3. Build a `FormulaGraph` over those input columns.
4. Wrap it in a thin function that calls `bind` + `graph.plan`.
5. Expose a public function only when it is tested.

```python
from schenberg.core.expr import exp
from schenberg.core.graph import FormulaGraph
from schenberg.market_data.roles import With, bind, market_role
from schenberg.market_data.snapshot import MarketSnapshot
import polars as pl

MyRate = (
    market_role("my_rate")
    .read("my_curves", "rate")
    .by(indexer="id_indexador", payment_days="tenor_days")
)

class MyInput(With[MyRate]):
    instrument_id: str
    indexer: str
    payment_days: int
    strike: float

_g = FormulaGraph("my_pricer", input=MyInput)
_c = _g.input
_T  = _g.let("year_fraction",  _c.payment_days / 252.0, symbol="T")
_DF = _g.let("discount_factor", exp(-_c.my_rate * _T),   symbol="DF")
_FV = _g.let("future_value",    _c.my_rate - _c.strike,  symbol="FV")
_PV = _g.let("present_value",   _FV * _DF,               symbol="PV")
_g.let("value", _PV, symbol="V")
_g.returns("output", future_value="future_value", present_value="present_value", value="value")

def price_my_instrument(trades: pl.LazyFrame, market: MarketSnapshot) -> pl.LazyFrame:
    enriched = bind(trades, market, MyInput)
    return _g.plan(enriched, view="output")
```

## Reuse the forward formula

Both generic and energy forwards share the same `FormulaGraph` structure — they
differ only in market roles. You can build a new forward variant by defining new
roles and a new input schema, without duplicating any formula logic:

```python
from schenberg.pricing.api import forward_formula
# forward_formula is a FormulaGraph; bind your own enriched frame to it:
result = forward_formula.plan(my_enriched_frame, view="output")
```

## Add a market fixing

A `Fixing` derives a date join-key from contract columns, used when the market
source is keyed by a fixing date rather than the tenor directly:

```python
from schenberg.market_data.roles import Fixing, market_role
import polars as pl

CpiFixing = (
    market_role("cpi_value")
    .read("fixings", "fixing_value")
    .by(indexer="id_indexador")
    .fixing(
        "fixing_date",
        Fixing.on("indexer")
            .when("CPI", pl.col("tenor") + pl.duration(days=5))
            .otherwise(pl.col("tenor")),
    )
)
```

## Market data

Use `MarketSource(..., unique_by=(...))` for quote tables whose keys must be
unique. Snapshot construction may collect those sources to validate metadata.
Avoid `.collect()` anywhere in pricing or position code.

## Add a position measure

Custom measures are small callables that register one term on a `PositionView`:

```python
from schenberg.position.view import Measure, PositionView
from schenberg.core.expr import Expr

def my_measure(*, name: str = "my_col") -> Measure:
    def register(view: PositionView) -> Expr:
        return view.let(name, view.col("exposure") * view.col("value") * 0.5)
    return Measure(register)

my_view = (
    PositionView("my_view", output=MyOutput)
    .spine(Position)
    .source("value", InstrumentValue, on=("instrument_type", "instrument_id"))
    .source("book", BookContract, on="book")
    .source("fx", ReportingFx, on=("currency", "reporting_currency"))
    .add(exposure(), mtm(), my_measure())
    .returns()
)
```

## Public API discipline

Do not document a public pricer until it is importable and tested from
`schenberg.pricing.api`. If an example imports a symbol, that symbol must exist.
