"""End-to-end portfolio pipeline on a near-real book of CDI-vs-IPCA swaps.

Exercises the whole stack: swap pricing -> position valuation -> book PnL across
two markets -> parallel DV01 under a bumped curve, and reconciles each report
against an independent computation.
"""

from __future__ import annotations

from typing import cast

import polars as pl
import pytest
from schenberg.pricing.portfolio import (
    bump_curve,
    compute_dv01,
    compute_pnl,
    make_valuer,
)

from .realistic_data import make_market, make_positions, make_swaps

CATALOG_SIZE = 40


@pytest.fixture(scope="module")
def book():
    swaps = make_swaps(CATALOG_SIZE)
    swap_ids = cast(pl.DataFrame, swaps.select("swap_id").collect())["swap_id"].to_list()
    positions = make_positions(swap_ids)
    value = make_valuer(swaps)
    return positions, value


def test_book_pnl_reconciles_to_position_level(book) -> None:
    positions, value = book
    today = make_market(zero_rate=0.10)
    prev = make_market(zero_rate=0.11)  # rates were higher yesterday

    report = cast(pl.DataFrame, compute_pnl(positions, today, prev, value=value).collect())

    # one row per book, all finite
    assert set(report["book"]) == {"Rates", "Inflation", "Macro"}
    assert report["pnl"].is_finite().all()

    # independent reconciliation: sum of book PnL == sum of position mv moves
    mv_today = cast(pl.DataFrame, value(positions, today).collect())["market_value"].sum()
    mv_prev = cast(pl.DataFrame, value(positions, prev).collect())["market_value"].sum()
    assert report["pnl"].sum() == pytest.approx(float(mv_today) - float(mv_prev), rel=1e-9)


def test_parallel_dv01_is_consistent_and_roughly_linear(book) -> None:
    positions, value = book
    market = make_market(zero_rate=0.10)

    one_bp = cast(
        pl.DataFrame,
        compute_dv01(positions, market, bump_curve(market, 0.0001), value=value).collect(),
    )
    two_bp = cast(
        pl.DataFrame,
        compute_dv01(positions, market, bump_curve(market, 0.0002), value=value).collect(),
    )

    assert set(one_bp["book"]) == {"Rates", "Inflation", "Macro"}
    assert one_bp["dv01"].is_finite().all()
    assert one_bp["dv01"].abs().sum() > 0.0  # a rates book has real sensitivity

    # reconciliation against base/bumped market values
    base_mv = cast(pl.DataFrame, value(positions, market).collect())["market_value"].sum()
    bumped_mv = cast(pl.DataFrame, value(positions, bump_curve(market, 0.0001)).collect())[
        "market_value"
    ].sum()
    assert one_bp["dv01"].sum() == pytest.approx(float(bumped_mv) - float(base_mv), rel=1e-9)

    # a 2bp parallel shift moves PV ~2x a 1bp shift (convexity is tiny at this scale)
    assert two_bp["dv01"].sum() == pytest.approx(2.0 * float(one_bp["dv01"].sum()), rel=0.02)


def test_pipeline_stays_lazy_until_collect(book) -> None:
    positions, value = book
    market = make_market()
    out = compute_pnl(positions, market, bump_curve(market, 0.0001), value=value)
    assert isinstance(out, pl.LazyFrame)  # nothing collected inside the pipeline
