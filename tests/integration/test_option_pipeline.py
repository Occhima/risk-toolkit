"""End-to-end option pipeline: surface -> generalized BSM graph -> Greeks.

Exercises the whole option stack on a 24-line book and reconciles every report
against an independent computation: the graph price against the numpy model, the
three Greek engines against each other, and put-call parity across the book.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import polars as pl
import pytest
from schenberg.math.black_scholes import GREEK_NAMES, generalized_price
from schenberg.pricing.instruments.option import price_options_with_greeks
from schenberg.pricing.instruments.option.models import option_price_router, option_risk_router

from .option_data import make_market, make_options


@pytest.fixture(scope="module")
def book():
    return make_options(), make_market()


def _eta(df: pl.DataFrame) -> np.ndarray:
    return np.where(df["option_kind"].to_numpy() == "CALL", 1.0, -1.0)


def test_graph_price_reconciles_to_numpy_model(book) -> None:
    options, market = book
    state = cast(
        pl.DataFrame,
        option_price_router.compute(options, market=market, view="state").collect(),
    )

    independent = generalized_price(
        state["spot"].to_numpy(), state["strike"].to_numpy(), state["rate"].to_numpy(),
        state["cost_of_carry"].to_numpy(), state["vol"].to_numpy(),
        state["year_fraction"].to_numpy(), _eta(state),
    )  # fmt: skip
    # graph uses an A&S normal CDF, the model an exact erf: agree to ~1e-6.
    assert np.allclose(state["price"].to_numpy(), independent, atol=1e-5)
    assert (state["price"].to_numpy() > 0).all()


def test_three_greek_backends_reconcile_across_the_book(book) -> None:
    options, market = book
    frames = {
        b: cast(pl.DataFrame, price_options_with_greeks(options, market, backend=b).collect()).sort(
            "option_id"
        )
        for b in ("CLOSED_FORM", "NUMERIC", "AUTODIFF")
    }
    a, n, d = frames["CLOSED_FORM"], frames["NUMERIC"], frames["AUTODIFF"]
    for name in GREEK_NAMES:
        # graph closed-form uses an A&S normal CDF, the numpy backends an exact
        # erf, so they agree to ~1e-6; finite differences are looser.
        assert np.allclose(a[name].to_numpy(), d[name].to_numpy(), rtol=1e-5, atol=1e-5), name
        assert np.allclose(a[name].to_numpy(), n[name].to_numpy(), rtol=1e-3, atol=1e-3), name


def test_book_respects_put_call_parity(book) -> None:
    options, market = book
    df = cast(
        pl.DataFrame,
        option_price_router.compute(options, market=market, view="state").collect(),
    )
    # pair each call with its put (same model, strike, maturity) and check
    # C - P = S e^{(b-r)T} - K e^{-rT}.
    ttm = pl.col("year_fraction")
    df = df.with_columns(
        carry_spot=pl.col("spot") * ((pl.col("cost_of_carry") - pl.col("rate")) * ttm).exp(),
        disc_strike=pl.col("strike") * (-pl.col("rate") * ttm).exp(),
    )
    pivot = df.pivot(
        on="option_kind", index=["option_model", "strike", "payment_days"], values="price"
    ).join(
        df.select("option_model", "strike", "payment_days", "carry_spot", "disc_strike").unique(),
        on=["option_model", "strike", "payment_days"],
    )
    parity = (pivot["CALL"] - pivot["PUT"]).to_numpy()
    expected = (pivot["carry_spot"] - pivot["disc_strike"]).to_numpy()
    assert np.allclose(parity, expected, atol=1e-5)


def test_risk_columns_have_sane_signs(book) -> None:
    options, market = book
    # The risk view satisfies exactly the OptionPriceWithGreeks contract, so
    # option_kind is recovered by joining back to the trades on option_id.
    risk = cast(
        pl.DataFrame,
        option_risk_router.compute(options, market=market, view="risk").collect(),
    )
    kinds = cast(pl.DataFrame, options.select("option_id", "option_kind").collect())
    df = risk.join(kinds, on="option_id")
    assert (df["gamma"].to_numpy() > 0).all()
    assert (df["vega"].to_numpy() > 0).all()
    calls = df.filter(pl.col("option_kind") == "CALL")
    puts = df.filter(pl.col("option_kind") == "PUT")
    assert (calls["delta"].to_numpy() > 0).all()
    assert (puts["delta"].to_numpy() < 0).all()


def test_pipeline_stays_lazy_until_collect(book) -> None:
    options, market = book
    assert isinstance(price_options_with_greeks(options, market), pl.LazyFrame)
