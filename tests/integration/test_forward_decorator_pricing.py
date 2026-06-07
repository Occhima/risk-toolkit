from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest
from schenberg import FormulaGraph, MarketSnapshot, With, bind, exp, market_role
from schenberg.domain.base import SchenbergDataFrameModel

ForwardRate = (
    market_role("forward_rate")
    .read("curves", "forward_rate")
    .by(indexer="id_indexador", payment_days="tenor_days")
)
RiskFreeRate = (
    market_role("risk_free_rate")
    .read("curves", "risk_free_rate")
    .by(indexer="id_indexador", payment_days="tenor_days")
)


class ForwardInput(With[ForwardRate], With[RiskFreeRate], SchenbergDataFrameModel):
    instrument_id: str
    indexer: str
    currency: str
    strike: float
    payment_days: int


def forward_graph() -> FormulaGraph:
    g = FormulaGraph("forward", input=ForwardInput)

    @g.formula(symbol="T")
    def year_fraction(c):
        return c.payment_days / 252.0

    @g.formula(symbol="DF")
    def discount_factor(c, year_fraction):
        return exp(-c.risk_free_rate * year_fraction)

    @g.formula(symbol="FV")
    def future_value(c):
        return c.forward_rate - c.strike

    @g.formula(symbol="PV")
    def present_value(future_value, discount_factor):
        return future_value * discount_factor

    @g.formula(symbol="Delta")
    def delta(discount_factor):
        return discount_factor

    g.returns(
        "output",
        instrument_id="instrument_id",
        future_value="future_value",
        present_value="present_value",
        value="present_value",
        delta="delta",
        currency="currency",
    )
    return g


def test_forward_decorator_pricing_with_market_bind_is_lazy_and_correct() -> None:
    trades = pl.DataFrame(
        {
            "instrument_id": ["FWD-1", "FWD-2"],
            "indexer": ["SOY", "SOY"],
            "currency": ["USD", "USD"],
            "strike": [1000.0, 1020.0],
            "payment_days": [252, 504],
        }
    ).lazy()
    market = (
        MarketSnapshot.at(date(2026, 6, 6))
        .source(
            "curves",
            pl.DataFrame(
                {
                    "id_indexador": ["SOY", "SOY"],
                    "tenor_days": [252, 504],
                    "forward_rate": [1050.0, 1100.0],
                    "risk_free_rate": [0.10, 0.12],
                }
            ),
            unique_by=("id_indexador", "tenor_days"),
        )
        .build()
    )
    graph = forward_graph()
    enriched = bind(trades, market, ForwardInput)
    priced = graph.plan(enriched, view="output")
    assert isinstance(priced, pl.LazyFrame)
    assert "side" not in graph.required_inputs("output")

    out = priced.collect()
    assert {"value", "delta"} <= set(out.columns)
    assert "side" not in out.columns
    rows = {row["instrument_id"]: row for row in out.to_dicts()}
    want_delta_1 = math.exp(-0.10 * 1.0)
    assert rows["FWD-1"]["future_value"] == pytest.approx(50.0)
    assert rows["FWD-1"]["value"] == pytest.approx(50.0 * want_delta_1)
    assert rows["FWD-1"]["delta"] == pytest.approx(want_delta_1)
    want_delta_2 = math.exp(-0.12 * 2.0)
    assert rows["FWD-2"]["value"] == pytest.approx(80.0 * want_delta_2)
    assert rows["FWD-2"]["delta"] == pytest.approx(want_delta_2)
