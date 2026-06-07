from __future__ import annotations

import math

import polars as pl
import pytest
from schenberg.core.expr import exp, var
from schenberg.core.graph import Formula
from schenberg.domain.base import SchenbergDataFrameModel


class ForwardPricingInput(SchenbergDataFrameModel):
    instrument_id: str
    strike: float
    payment_days: int
    forward_price: float
    risk_free: float


class ForwardPricing(SchenbergDataFrameModel):
    future_value: float
    present_value: float


def _forward() -> Formula:
    f = Formula[ForwardPricingInput, ForwardPricing]("forward")
    c = f.contract
    year_fraction = f.let("year_fraction", c.payment_days / 252.0, symbol="T")
    discount_factor = f.let("discount_factor", exp(-c.risk_free * year_fraction), symbol="DF")
    future_value = f.let("future_value", c.forward_price - c.strike, symbol="FV")
    f.let("present_value", future_value * discount_factor, symbol="PV")
    f.returns(ForwardPricing)
    return f


def _frame() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "instrument_id": ["A", "B"],
            "strike": [4.0, 5.0],
            "payment_days": [252, 504],
            "forward_price": [5.0, 6.0],
            "risk_free": [0.10, 0.12],
        }
    ).lazy()


def test_plan_prices_purely() -> None:
    out = _forward().plan(_frame()).collect()
    assert out.columns == ["future_value", "present_value"]
    rows = out.to_dicts()
    assert rows[0]["future_value"] == pytest.approx(1.0)
    assert rows[0]["present_value"] == pytest.approx(1.0 * math.exp(-0.10 * 252 / 252))
    assert rows[1]["present_value"] == pytest.approx(1.0 * math.exp(-0.12 * 504 / 252))


def test_required_inputs_excludes_derived_terms() -> None:
    f = _forward()
    assert f.required_inputs() == {"payment_days", "risk_free", "forward_price", "strike"}


def test_missing_input_fails_fast() -> None:
    bad = _frame().drop("risk_free")
    with pytest.raises(ValueError, match="missing required input"):
        _forward().plan(bad).collect()


def test_latex_is_derived_from_formula() -> None:
    f = _forward()
    latex = f.to_latex("present_value")
    assert latex.startswith("PV = ")
    assert f.formula_of("present_value") == latex
    assert f.formulas()["present_value"] == latex


def test_topological_order_is_dependency_sound() -> None:
    order = _forward().topological_order()
    assert order.index("year_fraction") < order.index("discount_factor")
    assert order.index("future_value") < order.index("present_value")


def test_self_reference_rejected() -> None:
    f = Formula[ForwardPricingInput, ForwardPricing]("bad")
    with pytest.raises(ValueError, match="references itself"):
        f.let("loop", var("loop") + 1.0)


def test_mermaid_includes_formula_type_and_outputs() -> None:
    diagram = _forward().to_mermaid()
    assert "Formula[ForwardPricingInput, ForwardPricing]" in diagram
    assert 'future_value --> future_value_out["future_value"]' in diagram
    assert 'present_value --> present_value_out["present_value"]' in diagram
