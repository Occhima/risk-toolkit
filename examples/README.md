# Examples

Runnable, self-contained scripts. Each builds its own market and inputs, prints
a result, and stays lazy until the final `.collect()`.

```bash
uv run python examples/01_price_a_swap.py
uv run python examples/02_energy_forward.py
uv run python -m examples.custom_instrument.run
```

| Example | Shows |
|---------|-------|
| [`01_price_a_swap.py`](01_price_a_swap.py) | Price a CDI-vs-IPCA swap through the public `price_swap` facade. |
| [`02_energy_forward.py`](02_energy_forward.py) | An energy forward reusing the generic forward backbone, sourcing its price from a curve. |
| [`custom_instrument/`](custom_instrument/) | **Extending the toolkit**: a brand-new inflation-linked energy forward — a custom graph + an index-convention registry. Start with its [README](custom_instrument/README.md). |

New to the concepts? Read [`docs/concepts.md`](../docs/concepts.md) first.
