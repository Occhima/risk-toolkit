# Examples

Runnable, self-contained scripts. Each builds its own market snapshot and
trades frame, and stays lazy until the final `.collect()`.

```bash
# run from the repo root
uv run python docs/examples/01_price_a_forward.py
uv run python docs/examples/02_energy_forward.py
uv run python docs/examples/03_inspect_a_formula.py
uv run python docs/examples/04_shock_scenario.py
uv run python -m docs.examples.custom_instrument.run
```

## Quick map

| Example | What it shows |
|---------|---------------|
| [01 — Generic forward](01_price_a_forward.md) | `price_forward` with `ForwardContractPricing`; contract rules fill `index_fixing_date` / `currency_fixing_date` automatically. |
| [02 — Energy forward](02_energy_forward.md) | `price_energy_forward` with `EnergyForwardPricing`; PLD fixing-date override; energy curve + DI discounting + FX. |
| [03 — Inspect a formula](03_inspect_a_formula.md) | `explain()`, `to_mermaid()`, `info()`, `required_inputs()`, `dependencies_of()` — the same declaration that prices also documents itself. |
| [04 — Shock scenario](04_shock_scenario.md) | `MarketPath` + `Shock.compose` to reprice under a parallel rate bump; immutability of the original market. |
| [Custom instrument](custom_instrument/index.md) | **Extend the toolkit**: a brand-new inflation-linked energy forward — custom graph + index-convention registry. |

## How pricing works

```
trades (LazyFrame)          market (MarketSnapshot)
       │                           │
       └──────── price_*(…) ───────┘
                     │
                     ▼  lazy
              LazyFrame[OutputSchema]
                     │
                     ▼  .collect()  ← you decide when
              DataFrame
```

The pricing functions are annotated with `@pa.check_types(lazy=True)`.
[Contract rules](../concepts.md#contracts-at-the-boundary) — encoded with
`@rule_for` on the schema class — run before Pandera validates, filling any
derived date columns (`index_fixing_date`, `currency_fixing_date`) from the
contract terms. You never call `.resolve()` or `.collect()` yourself inside
a pricing function.

## New to the concepts?

Read [Concepts](../concepts.md) for the mental model, then come back here for
runnable code. [Extending](../extending.md) explains how to add your own
instrument or market dimension.
