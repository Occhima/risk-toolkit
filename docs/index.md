# Schenberg Risk Toolkit

**Composable, lazy pricing for financial instruments — as a graph of formulas.**

Schenberg is a lazy, contract-oriented pricing DSL built on lazy [Polars](https://pola.rs)
dataframes, [Pandera](https://pandera.readthedocs.io) boundary schemas, and a small graph
engine (`rustworkx`). Inputs, market reads and formulas are **Terms** inside a
**FormulaGraph**; the `MarketSnapshot` is the environment supplied at compute time. The
same graph declaration can be interpreted as lazy Polars, a Mermaid diagram, explanation
text, or debug stages, and never collects until you ask.

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
| **[Concepts](concepts.md)** | Mental model: Term, FormulaGraph, Router as ArrowChoice, MarketSnapshot, Workflow |
| **[Extending](extending.md)** | Add an instrument, a variant, or a market dimension |
| **[API Reference](api/index.md)** | Auto-generated from source docstrings |
