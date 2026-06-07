from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from schenberg import CURVES, FIXINGS, VOLS, MarketSnapshot, With, bind, market_role
from schenberg.domain.base import SchenbergDataFrameModel
from schenberg.market_data.date_rules import same_day
from schenberg.market_data.roles import Fixing, MarketRole


def test_semantic_roles_build_market_roles() -> None:
    role = (
        CURVES.zero_rate("BRL_DI", as_="risk_free_rate").source("curves").for_tenor("payment_days")
    )
    assert isinstance(role, MarketRole)
    assert role.name == "risk_free_rate"
    assert role.source == "curves"
    assert [(b.left, b.right) for b in role.exact] == [("payment_days", "tenor_days")]
    assert [(b.value, b.right) for b in role.literal] == [("BRL_DI", "curve")]


def test_semantic_expiry_strike_and_fixing() -> None:
    vol = (
        VOLS.implied("USD/BRL", as_="vol")
        .source("vol_surface")
        .for_expiry("expiry")
        .for_strike("strike")
    )
    assert [(b.left, b.right) for b in vol.exact] == [
        ("expiry", "expiry"),
        ("strike", "strike"),
    ]
    assert [(b.value, b.right) for b in vol.literal] == [("USD/BRL", "currency_pair")]
    role_expr = (
        FIXINGS.value(as_="spot")
        .source("fixings")
        .by(currency_pair="currency_pair")
        .fixing("fixing_date", same_day("pricing_date"))
    )
    assert isinstance(role_expr.fixing_rule, Fixing)
    role_fix = (
        FIXINGS.value(as_="spot")
        .source("fixings")
        .fixing("fixing_date", Fixing.rule(same_day("pricing_date")))
    )
    assert isinstance(role_fix.fixing_rule, Fixing)


def test_semantic_bind_matches_manual() -> None:
    SemanticRate = (
        CURVES.zero_rate("BRL_DI", as_="risk_free_rate").source("curves").for_tenor("payment_days")
    )
    ManualRate = (
        market_role("risk_free_rate")
        .read("curves", "zero_rate")
        .by(curve="curve", payment_days="tenor_days")
    )

    class SemanticInput(With[SemanticRate], SchenbergDataFrameModel):
        instrument_id: str
        curve: str
        payment_days: int

    class ManualInput(With[ManualRate], SchenbergDataFrameModel):
        instrument_id: str
        curve: str
        payment_days: int

    trades = pl.DataFrame(
        {"instrument_id": ["A"], "curve": ["BRL_DI"], "payment_days": [252]}
    ).lazy()
    market = (
        MarketSnapshot.at(date(2026, 6, 6))
        .source(
            "curves",
            pl.DataFrame({"curve": ["BRL_DI"], "tenor_days": [252], "zero_rate": [0.05]}),
            unique_by=("curve", "tenor_days"),
        )
        .build()
    )
    assert (
        bind(trades, market, SemanticInput).collect().to_dicts()
        == bind(trades, market, ManualInput).collect().to_dicts()
    )


def test_curve_name_argument_is_literal_join_key_not_trade_column() -> None:
    RiskFree = (
        CURVES.zero_rate("BRL_DI", as_="risk_free_rate").source("curves").for_tenor("payment_days")
    )

    class Input(With[RiskFree], SchenbergDataFrameModel):
        instrument_id: str
        payment_days: int

    trades = pl.DataFrame({"instrument_id": ["T1"], "payment_days": [252]}).lazy()

    market = (
        MarketSnapshot.at(date(2026, 6, 6))
        .source(
            "curves",
            pl.DataFrame(
                {
                    "curve": ["BRL_DI"],
                    "tenor_days": [252],
                    "zero_rate": [0.10],
                }
            ),
            unique_by=("curve", "tenor_days"),
        )
        .build(validate=False)
    )

    out = bind(trades, market, Input).collect()

    assert out["risk_free_rate"][0] == pytest.approx(0.10)
    assert "curve" not in out.columns
    assert not any(c.startswith("__const_") for c in out.columns)


def test_dynamic_curve_column_still_supported_with_by() -> None:
    RiskFree = (
        CURVES.zero_rate(as_="risk_free_rate")
        .source("curves")
        .by(curve="curve")
        .for_tenor("payment_days")
    )

    class Input(With[RiskFree], SchenbergDataFrameModel):
        instrument_id: str
        curve: str
        payment_days: int

    trades = pl.DataFrame(
        {"instrument_id": ["T1", "T2"], "curve": ["BRL_DI", "USD_SOFR"], "payment_days": [252, 252]}
    ).lazy()

    market = (
        MarketSnapshot.at(date(2026, 6, 6))
        .source(
            "curves",
            pl.DataFrame(
                {
                    "curve": ["BRL_DI", "USD_SOFR"],
                    "tenor_days": [252, 252],
                    "zero_rate": [0.10, 0.03],
                }
            ),
            unique_by=("curve", "tenor_days"),
        )
        .build(validate=False)
    )

    out = bind(trades, market, Input).collect()

    assert out["risk_free_rate"].to_list() == pytest.approx([0.10, 0.03])
    assert "curve" in out.columns
