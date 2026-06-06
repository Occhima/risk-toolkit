# Examples

Runnable, self-contained scripts and interactive Marimo notebooks. Each builds
its own market snapshot and trades frame, and stays lazy until the final
`.collect()`.

```bash
# Plain scripts — run from the repo root
uv run python docs/examples/01_price_a_forward.py
uv run python docs/examples/02_energy_forward.py
uv run python docs/examples/03_inspect_a_formula.py
uv run python docs/examples/04_shock_scenario.py

# Interactive notebooks
uv run marimo edit docs/examples/05_soja_forward.py
uv run marimo edit docs/examples/06_position_view.py
uv run marimo edit docs/examples/07_usdbrl_book_valuation.py

# Export notebooks to HTML (for offline viewing)
marimo export html docs/examples/05_soja_forward.py -o docs/examples/05_soja_forward.html
marimo export html docs/examples/06_position_view.py -o docs/examples/06_position_view.html
marimo export html docs/examples/07_usdbrl_book_valuation.py -o docs/examples/07_usdbrl_book_valuation.html
```

## Quick map

| # | Example | What it shows |
|---|---------|---------------|
| [01](01_price_a_forward.md) | Generic forward | `price_forward` with `ForwardContractPricing`; market join via `With[role]` + `bind`. |
| [02](02_energy_forward.md) | Energy forward | `price_energy_forward`; energy curve + DI discounting; same formula, different market roles. |
| [03](03_inspect_a_formula.md) | Inspect a formula | `explain()`, `to_mermaid()`, `info()`, `required_inputs()`, `dependencies_of()` on a live graph. |
| [04](04_shock_scenario.md) | Shock scenario | `MarketPath` + `Shock.compose` to reprice under a parallel rate bump; immutability of the original market. |
| 05 | Soja forward (Marimo) | Notebook interativo: pricing de forward CBOT NOV27, delta pelo preço da soja, mapas de sensibilidade. |
| 06 | Position view (Marimo) | View de posição com MTM, exposição e delta para um book de soja; `position_value` + `position_risk`. |
| 07 | USD/BRL NDF book (Marimo) | Book de NDF com MTM, PnL e PnL explain (roll/curve/fx waterfall). |

## How pricing works

```
trades (LazyFrame)          market (MarketSnapshot)
       │                           │
       └──── bind(trades, market, Schema) ────┘
                     │  joins market roles
                     ▼
              enriched LazyFrame
                     │
                     ▼
              formula.plan(enriched, view="output")
                     │  compiles Expr tree → with_columns
                     ▼  lazy
              LazyFrame  ← collect() when you need it
```

`bind` discovers the `With[role]` mixins on the input schema and executes a
left-join per role against the market snapshot. The formula graph then runs as a
pure `with_columns` — it never sees the snapshot.

## New to the concepts?

Read [Concepts](../concepts.md) for the mental model, then come back here for
runnable code. [Extending](../extending.md) shows how to add your own instrument
or market dimension.
