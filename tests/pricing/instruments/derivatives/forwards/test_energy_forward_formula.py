from __future__ import annotations

from schenberg.pricing.instruments.derivatives.forwards.energy import (
    energy_forward_formula,
)


def test_energy_forward_reuses_forward_formula_terms() -> None:
    info = energy_forward_formula.info(view="output")

    assert "forward_price" in info.market_outputs
    assert "risk_free" in info.market_outputs
    assert "currency" in info.market_outputs
    assert "future_value" in info.formula_nodes
    assert "present_value" in info.formula_nodes
    assert "value" in info.formula_nodes
