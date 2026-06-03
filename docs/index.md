# Schenberg Risk Toolkit

**Composable, lazy pricing for financial instruments — as a graph of formulas.**

Schenberg is a Python pricing toolkit built on lazy [Polars](https://pola.rs) dataframes,
[Pandera](https://pandera.readthedocs.io) boundary schemas, and a small graph engine
(`rustworkx`) for composable pricing formulas. You describe a price as a DAG of
row-local formulas; the engine compiles it into a single lazy expression and never
collects until you ask.

```python
from datetime import date
import polars as pl

from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.api import price_swap

market = MarketSnapshot.from_sources(
    as_of=date.today(),
    sources=[
        MarketSource("curves", pl.DataFrame({
            "id_indexador": [1, 2], "tenor_days": [252, 252],
            "zero_rate": [0.10, 0.05], "forward_rate": [0.12, None],
        })),
    ],
)
```

## Where to start

| | |
|---|---|
| **[Concepts](concepts.md)** | Mental model: FormulaGraph, Router, MarketSnapshot, Workflow |
| **[Extending](extending.md)** | Add an instrument, a variant, or a market dimension |
| **[API Reference](api/index.md)** | Auto-generated from source docstrings |
