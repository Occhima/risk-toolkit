"""Inspect a pricing formula without pricing anything.

Run with:  uv run python docs/examples/03_inspect_a_formula.py

The same declaration that computes prices also explains itself: required inputs,
market reads, formula nodes, output columns, a Mermaid flowchart, and a
dependency trace. None of this ever calls ``collect`` — the graph is a
program you can read before running.

This is the key property of the declarative DSL: every piece of pricing logic
is statically inspectable at the formula level, not just at the data level.
"""

from __future__ import annotations

from schenberg.pricing.instruments.derivatives.forwards.energy.api import (
    energy_forward_formula,
)
from schenberg.pricing.instruments.derivatives.forwards.formulas import forward_formula

# ---------------------------------------------------------------------------
# 1. What does the generic forward need from the caller?
# ---------------------------------------------------------------------------
print("=== Generic forward — required contract inputs ===")
print(sorted(forward_formula.required_inputs()))
# ['currency', 'indexer', 'payment_days', 'strike']
# (plus tenor / instrument_id from the contract schema — those are also inputs)

# ---------------------------------------------------------------------------
# 2. Full explain: inputs, market reads, formula path, returns
# ---------------------------------------------------------------------------
print("\n=== Generic forward — explain(view='output') ===")
print(forward_formula.explain(view="output"))

# ---------------------------------------------------------------------------
# 3. GraphInfo: machine-readable version of the same data
# ---------------------------------------------------------------------------
info = forward_formula.info(view="output")
print("\n=== GraphInfo fields ===")
print("market_outputs :", info.market_outputs)
print("formula_nodes  :", info.formula_nodes)

# ---------------------------------------------------------------------------
# 4. Mermaid diagram — paste into https://mermaid.live to visualise
# ---------------------------------------------------------------------------
print("\n=== Mermaid flowchart ===")
print(forward_formula.to_mermaid(view="output"))

# ---------------------------------------------------------------------------
# 5. Dependency trace: what does 'present_value' depend on?
# ---------------------------------------------------------------------------
print("=== Dependencies of present_value ===")
print(sorted(forward_formula.dependencies_of("present_value")))

# ---------------------------------------------------------------------------
# 6. Energy forward inherits the same formula — only the market differs
# ---------------------------------------------------------------------------
print("\n=== Energy forward — explain(view='output') ===")
print(energy_forward_formula.explain(view="output"))

print("\n=== Energy forward — Mermaid ===")
print(energy_forward_formula.to_mermaid(view="output"))
