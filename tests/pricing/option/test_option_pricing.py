from __future__ import annotations

import math
from typing import cast

import polars as pl
import pytest
from schenberg.core.graph import ExprGraph
from schenberg.market_data.interpolated import InterpolatedBook
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.instruments.option import price_options, price_options_with_greeks
from schenberg.pricing.instruments.option.models import option_price_router, option_router


def _priced(option_inputs, option_market) -> pl.DataFrame:
    return cast(pl.DataFrame, price_options(option_inputs, option_market).collect()).sort(
        "option_id"
    )


def test_router_covers_the_four_model_kind_leaves() -> None:
    expected_leaves = 4  # {GENERALIZED, MERTON} x {CALL, PUT}
    assert len(option_router.cases) == expected_leaves


def test_generalized_call_matches_textbook_black_scholes(option_inputs, option_market) -> None:
    # b = r = 0.10, vol = 0.20, S = K = 100, T = 1 -> plain BS 1973.
    df = _priced(option_inputs, option_market)
    row = df.filter(pl.col("option_id") == "G-C")
    d1 = (0.10 + 0.5 * 0.20**2) / 0.20
    expected = 100 * _ncdf(d1) - 100 * math.exp(-0.10) * _ncdf(d1 - 0.20)
    assert row["price"].item() == pytest.approx(expected, abs=1e-4)
    assert df.columns == ["option_id", "instrument_type", "price"]


def test_put_call_parity_generalized(option_inputs, option_market) -> None:
    df = _priced(option_inputs, option_market)
    call = df.filter(pl.col("option_id") == "G-C")["price"].item()
    put = df.filter(pl.col("option_id") == "G-P")["price"].item()
    # carry = rate -> forward factor is 1: C - P = S - K e^{-rT}
    assert call - put == pytest.approx(100 - 100 * math.exp(-0.10), abs=1e-4)


def test_put_call_parity_merton_uses_dividend_yield(option_inputs, option_market) -> None:
    df = _priced(option_inputs, option_market)
    call = df.filter(pl.col("option_id") == "M-C")["price"].item()
    put = df.filter(pl.col("option_id") == "M-P")["price"].item()
    # Merton: C - P = S e^{-qT} - K e^{-rT}, q = 0.03
    assert call - put == pytest.approx(100 * math.exp(-0.03) - 100 * math.exp(-0.10), abs=1e-4)


def test_merton_equals_generalized_with_carry_minus_dividend(option_inputs, option_market) -> None:
    # Re-point the carry curve to b = r - q; GENERALIZED must then equal MERTON.
    carry = pl.DataFrame(
        {"id_indexador": [1, 1, 1], "tenor_days": [126, 252, 504], "cost_of_carry": [0.07] * 3}
    ).lazy()
    rqmarket = option_market.with_source(MarketSource("carry_curve", carry))
    df = _priced(option_inputs, rqmarket)
    gen_call = df.filter(pl.col("option_id") == "G-C")["price"].item()
    mer_call = df.filter(pl.col("option_id") == "M-C")["price"].item()
    assert gen_call == pytest.approx(mer_call, abs=1e-9)


def test_vol_is_interpolated_off_the_surface(option_market) -> None:
    book = InterpolatedBook.from_quotes(
        option_market.source("vol_surface").data,
        group_col="id_indexador",
        axis_cols=("tenor_days", "strike"),
        value_col="implied_vol",
    )
    # on a grid node: exact quote (252d, K=100) = 0.20
    assert book.interpolate([1], [252], [100.0])[0] == pytest.approx(0.20)
    # between tenors and strikes: strictly inside the bracketing quotes (all 0.19..0.25)
    surface_lo, surface_hi = 0.19, 0.25
    v = book.interpolate([1], [189], [95.0])[0]
    assert surface_lo < v < surface_hi


def test_pricing_stays_lazy_until_collect(option_inputs, option_market) -> None:
    out = price_options(option_inputs, option_market)
    assert isinstance(out, pl.LazyFrame)


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def test_option_graph_declares_vol_surface() -> None:
    graph = cast(ExprGraph, option_price_router.cases[0][1])
    assert "vol" in graph.info(output_profile="price").market_outputs


def test_price_options_with_greeks_returns_public_columns(option_inputs, option_market) -> None:
    df = cast(pl.DataFrame, price_options_with_greeks(option_inputs, option_market).collect())
    assert df.columns == [
        "option_id",
        "instrument_type",
        "price",
        "delta",
        "gamma",
        "vega",
        "theta",
        "rho",
    ]
