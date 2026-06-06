"""Inspect a pricing formula without pricing anything.

Run with:  uv run python docs/examples/03_inspect_a_formula.py

The same declaration that computes prices also explains itself: required inputs,
formula nodes in dependency order, a Mermaid flowchart, and a dependency trace.
None of this ever calls ``collect`` — the graph is a program you can read before
running.

This is the key property of the declarative DSL: every piece of pricing logic is
statically inspectable at the formula level, not just at the data level.
"""

from __future__ import annotations

from schenberg.pricing.api import energy_forward_formula, forward_formula

# ---------------------------------------------------------------------------
# 1. What does the generic forward need from the caller?
# ---------------------------------------------------------------------------
print("=== Generic forward — required inputs for 'output' view ===")
print(sorted(forward_formula.required_inputs("output")))
# ['forward_rate', 'payment_days', 'risk_free_rate', 'strike']

# ---------------------------------------------------------------------------
# 2. Full explain: inputs, formula path, returns
# ---------------------------------------------------------------------------
print("\n=== Generic forward — explain(view='output') ===")
print(forward_formula.explain(view="output"))

# ---------------------------------------------------------------------------
# 3. GraphInfo: machine-readable version of the same data
# ---------------------------------------------------------------------------
info = forward_formula.info(view="output")
print("\n=== GraphInfo fields ===")
print("required_inputs  :", info.required_inputs)
print("formula_nodes    :", info.formula_nodes)
print("intermediate_nodes:", info.intermediate_nodes)
print("view_nodes       :", info.view_nodes)

# ---------------------------------------------------------------------------
# 4. Mermaid diagram — paste into https://mermaid.live to visualise
# ---------------------------------------------------------------------------
print("\n=== Mermaid flowchart ===")
print(forward_formula.to_mermaid())

# ---------------------------------------------------------------------------
# 5. Dependency trace: what does 'present_value' depend on?
# ---------------------------------------------------------------------------
print("\n=== Dependencies of present_value ===")
print(sorted(forward_formula.dependencies_of("present_value")))

# ---------------------------------------------------------------------------
# 6. LaTeX rendering — the formulas as written, not reconstructed
# ---------------------------------------------------------------------------
print("\n=== Formula strings ===")
for _name, formula in forward_formula.formulas().items():
    print(f"  {formula}")

# ---------------------------------------------------------------------------
# 7. Energy forward — same formula, different market roles
# ---------------------------------------------------------------------------
print("\n=== Energy forward — explain(view='output') ===")
print(energy_forward_formula.explain(view="output"))
