from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest
from schenberg import Fixing, FormulaGraph, MarketSnapshot, With, bind, exp, market_role
from schenberg.domain.base import SchenbergDataFrameModel
from schenberg.market_data.date_rules import previous_business_days

Ptax = (
    market_role("ptax")
    .read("ptax_fixings", "fixing_value")
    .by(currency_pair="currency_pair")
    .fixing("ptax_fixing_date", Fixing.rule(previous_business_days("tenor", n=5)))
)
RiskFree = (
    market_role("risk_free_rate").read("curves", "risk_free_rate").by(payment_days="tenor_days")
)


class UsdBrlDfInput(With[Ptax], With[RiskFree], SchenbergDataFrameModel):
    instrument_id: str
    currency_pair: str
    currency: str
    tenor: date
    ptax_fixing_date: date
    contracted_rate: float
    notional_usd: float
    payment_days: int


def usdbrl_graph() -> FormulaGraph:
    g = FormulaGraph("usdbrl_df", input=UsdBrlDfInput)

    @g.formula(symbol="T")
    def year_fraction(c):
        return c.payment_days / 252.0

    @g.formula(symbol="DF")
    def discount_factor(c, year_fraction):
        return exp(-c.risk_free_rate * year_fraction)

    @g.formula(symbol="PV")
    def present_value(c, discount_factor):
        return c.notional_usd * (c.contracted_rate - c.ptax) * discount_factor

    g.returns(
        "output",
        instrument_id="instrument_id",
        ptax_fixing_date="ptax_fixing_date",
        ptax="ptax",
        value="present_value",
        currency="currency",
    )
    return g


def raw_trades() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "instrument_id": ["NDF-1"],
            "currency_pair": ["USD/BRL"],
            "currency": ["BRL"],
            "tenor": [date(2026, 6, 15)],
            "contracted_rate": [5.50],
            "notional_usd": [1_000_000.0],
            "payment_days": [21],
        }
    ).lazy()


def prepared_trades() -> pl.LazyFrame:
    return raw_trades().with_columns(
        Fixing.rule(previous_business_days("tenor", n=5)).expr().alias("ptax_fixing_date")
    )


def snapshot() -> MarketSnapshot:
    return (
        MarketSnapshot.at(date(2026, 6, 6))
        .source(
            "ptax_fixings",
            pl.DataFrame(
                {
                    "currency_pair": ["USD/BRL", "USD/BRL"],
                    "ptax_fixing_date": [date(2026, 6, 8), date(2026, 6, 9)],
                    "fixing_value": [5.37, 5.39],
                }
            ),
            unique_by=("currency_pair", "ptax_fixing_date"),
        )
        .source(
            "curves",
            pl.DataFrame({"tenor_days": [21], "risk_free_rate": [0.12]}),
            unique_by=("tenor_days",),
        )
        .build()
    )


def test_ptax_fixing_date_is_fifth_business_day_before_tenor() -> None:
    out = prepared_trades().select("ptax_fixing_date").collect()
    assert out.item() == date(2026, 6, 8)


def test_ptax_join_and_pricing_stay_lazy_until_collect() -> None:
    enriched = bind(prepared_trades(), snapshot(), UsdBrlDfInput)
    graph = usdbrl_graph()
    priced = graph.plan(enriched, view="output")
    staged = graph.stage(enriched, view="output")
    assert isinstance(priced, pl.LazyFrame)
    assert isinstance(staged, pl.LazyFrame)

    stage_columns = set(staged.collect_schema().names())
    assert "ptax_fixing_date" in stage_columns

    out = priced.collect()
    row = out.row(0, named=True)
    assert row["ptax_fixing_date"] == date(2026, 6, 8)
    assert row["ptax"] == pytest.approx(5.37)
    expected = 1_000_000.0 * (5.50 - 5.37) * math.exp(-0.12 * 21 / 252)
    assert row["value"] == pytest.approx(expected)
