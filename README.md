<div align="center">

# Schenberg Risk Toolkit

**Lazy pricing DSL built on symbolic formulas, Polars expressions, and typed boundary schemas.**

[Concepts](docs/concepts.md) · [Extending](docs/extending.md) · [Examples](docs/examples/)

</div>

Schenberg is a compact pricing DSL. Market data is resolved into ordinary input
columns before a formula graph runs; the graph itself is a pure symbolic program.
Pricing functions return lazy frames and do not execute trade-side queries until
the caller collects.

## Minimal forward example

```python
from datetime import date
import polars as pl

from schenberg import FormulaGraph, MarketSnapshot, With, bind, exp, market_role
from schenberg.domain.base import SchenbergDataFrameModel

ForwardRate = (
    market_role("forward_rate")
    .read("curves", "forward_rate")
    .by(indexer="id_indexador", payment_days="tenor_days")
)
RiskFreeRate = (
    market_role("risk_free_rate")
    .read("curves", "risk_free_rate")
    .by(indexer="id_indexador", payment_days="tenor_days")
)

class ForwardInput(With[ForwardRate], With[RiskFreeRate], SchenbergDataFrameModel):
    instrument_id: str
    indexer: str
    currency: str
    strike: float
    payment_days: int

g = FormulaGraph("forward", input=ForwardInput)

@g.formula(symbol="T")
def year_fraction(c):
    return c.payment_days / 252.0

@g.formula(symbol="DF")
def discount_factor(c, year_fraction):
    return exp(-c.risk_free_rate * year_fraction)

@g.formula(symbol="FV")
def future_value(c):
    return c.forward_rate - c.strike

@g.formula(symbol="PV")
def present_value(future_value, discount_factor):
    return future_value * discount_factor

@g.formula(symbol="Delta")
def delta(discount_factor):
    return discount_factor

g.returns(
    "output",
    instrument_id="instrument_id",
    future_value="future_value",
    present_value="present_value",
    value="present_value",
    delta="delta",
    currency="currency",
)

trades = pl.DataFrame({
    "instrument_id": ["FWD-1"],
    "indexer": ["DI"],
    "currency": ["BRL"],
    "strike": [100.0],
    "payment_days": [252],
}).lazy()
market = (
    MarketSnapshot.at(date(2026, 6, 6))
    .source(
        "curves",
        pl.DataFrame({
            "id_indexador": ["DI"],
            "tenor_days": [252],
            "forward_rate": [112.0],
            "risk_free_rate": [0.10],
        }),
        unique_by=("id_indexador", "tenor_days"),
    )
    .build()
)

result = g.plan(bind(trades, market, ForwardInput), view="output")  # LazyFrame
print(result.collect())
```

## Formula decorator

`@g.formula(...)` is the ergonomic API for registering terms. The default term
name is the Python function name; `name=`, `symbol=`, `description=`, `tags=` and
`dtype=` are supported. Dependencies are inferred from the function signature:
`c`, `contract`, `input` and `inputs` receive the graph input namespace, and any
parameter named after an earlier term receives that symbolic term.

The decorated function returns Schenberg `Expr` nodes, not opaque Python UDFs, so
introspection and compilation still work:

- `graph.formulas()` and `graph.formula_of(...)` derive LaTeX from the symbolic IR.
- `graph.explain(...)`, `graph.to_mermaid(...)`, `graph.info(...)`, and
  `graph.stage(...)` remain available.
- `graph.plan(...)` compiles to lazy Polars expressions and does not call
  `.collect()`.
- `g.let(...)` remains the lower-level primitive for manually registering a term.

## Position layer

Pricing returns pure per-instrument values and risks: no `side`, no book, no
comprado/vendido. The position layer lifts those pure values onto a `Position`:

```python
from schenberg.position import position_value, position_risk, book_value_rollup

pv = position_value(positions, value=instrument_values, book=book, fx=fx)
pr = position_risk(positions, risk=instrument_risk)
rollup = book_value_rollup.compute(pv)
```

## Design rules

- Market joins are declared as `With[role]` mixins and resolved by `bind` before
  the formula graph runs.
- Pure pricing graphs compute own-currency instrument values and sensitivities
  only; position direction belongs to the position layer.
- Missing inputs fail loudly at plan time.
- Examples define their pricers locally with the public API; Schenberg does not
  centralise those example pricers in a pricing API module.
- HTML examples are exported directly with `marimo export html`.

## Interactive examples

```bash
uv run marimo edit docs/examples/01_forward_pricer.py
uv run marimo edit docs/examples/02_forward_positions.py
uv run marimo edit docs/examples/03_usdbrl_df_fixing.py
uv run poe examples-html
```

## Install and check

```bash
uv sync --all-groups
uv run pytest
uv run poe check
```

## Semantic market roles and option example

`FormulaGraph` sees only resolved parameters. Semantic helpers such as `CURVES`,
`FIXINGS`, and `VOLS` build `MarketRole` declarations outside the graph; `bind(...)`
uses those roles to enrich a trade frame, and `graph.plan(...)` prices lazily.
Pandera validation belongs at public pricer/schema boundaries (for example via
`@price_function`), not inside formula math.

```python
from schenberg import CURVES, FIXINGS, VOLS, FormulaGraph, With, bind

Spot = FIXINGS.value("USD/BRL", as_="spot").source("fixings").by(
    currency_pair="currency_pair"
)
Vol = (
    VOLS.implied("USD/BRL", as_="vol")
    .source("vol_surface")
    .for_expiry("expiry")
    .for_strike("strike")
)
RiskFree = CURVES.zero_rate("BRL_DI", as_="risk_free_rate").source("curves").for_tenor(
    "payment_days"
)

class VanillaOptionInput(With[Spot], With[Vol], With[RiskFree], SchenbergDataFrameModel):
    ...

enriched = bind(trades, market, VanillaOptionInput)
priced = option_graph.plan(enriched, view="output")  # LazyFrame
```

See `docs/examples/04_vanilla_option.py` for Black-Scholes price/Greeks and
`docs/examples/05_option_pnl_explain.py` for a small lazy repricing PnL explain.
