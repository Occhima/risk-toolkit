# Examples

Runnable, self-contained scripts. Each builds its own market and inputs, prints
a result, and stays lazy until the final `.collect()`.

```bash
uv run python examples/01_price_a_swap.py
uv run python examples/02_energy_forward.py
uv run python examples/03_price_an_option.py
uv run python examples/04_shock_scenario.py
uv run python -m examples.custom_instrument.run

# Interactive marimo notebook (uv sync --group dev first):
uv run marimo edit examples/notebooks/energy_forward.py
```

| Example | Shows |
|---------|-------|
| [`01_price_a_swap.py`](01_price_a_swap.py) | Price a CDI-vs-IPCA swap; the swap `Structure` applies `leg_weight` and folds by `swap_id`. |
| [`02_energy_forward.py`](02_energy_forward.py) | An energy forward reusing the generic forward backbone, sourcing its price from a curve. |
| [`03_price_an_option.py`](03_price_an_option.py) | A generalized Black-Scholes-Merton option priced off an interpolated vol surface. |
| [`04_shock_scenario.py`](04_shock_scenario.py) | Reprice under a `Shock` built via `MarketPath`; inspect the swap `Structure` with `explain()`. |
| [`custom_instrument/`](custom_instrument/) | **Extending the toolkit**: a brand-new inflation-linked energy forward — a custom graph + an index-convention registry. Start with its [README](custom_instrument/README.md). |
| [`notebooks/energy_forward.py`](notebooks/energy_forward.py) | **Interactive** marimo notebook: the energy-forward `PricingGraph` end to end — requirements, the formula graph, a mermaid diagram, and a live strike slider. |

New to the concepts? Read [`docs/concepts.md`](../docs/concepts.md) first.
