"""Portfolio layer: book valuation, PnL, and risk composed over the pricing atom.

These are plain functions (not a Pipeline class): valuation is the reusable atom,
and PnL / risk are higher-order functions that inject a `value` callable. Every
boundary is a Pandera contract. Everything stays lazy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.core.graph import ExprGraph
from schenberg.core.market import MarketSnapshot
from schenberg.domain.schemas import SwapInput
from schenberg.pricing.instruments.swap.engine import price_swap

# ---------------------------------------------------------------------------
# PnL attribution as an ExprGraph (row-local formula math). Decompose the day's
# move into carry + market; total must reconcile to pv_today - pv_prev. Use
# pnl_attribution_graph.stage(...) for debugging which component breaks.
# ---------------------------------------------------------------------------
pnl_attribution_graph = ExprGraph("pnl_attribution")


@pnl_attribution_graph.node(tags=("pnl", "carry"))
def carry_pnl(pv_carry, pv_prev):  # roll-forward on yesterday's curve
    return pv_carry - pv_prev


@pnl_attribution_graph.node(tags=("pnl", "market"))
def market_pnl(pv_today, pv_carry):  # today's curve vs the rolled-forward
    return pv_today - pv_carry


@pnl_attribution_graph.node(tags=("pnl",))
def total_pnl(carry_pnl, market_pnl):
    return carry_pnl + market_pnl


pnl_attribution_graph.with_outputs(
    "attribution", carry_pnl="carry_pnl", market_pnl="market_pnl", total_pnl="total_pnl"
)


# ---------------------------------------------------------------------------
# Boundary contracts: every pipeline edge is typed.
# ---------------------------------------------------------------------------
class Positions(pa.DataFrameModel):
    """Portfolio holdings referencing instruments (NOT the instrument economics)."""

    class Config:
        coerce = True

    position_id: str
    book: str
    swap_id: str  # reference into the instrument catalog (SwapInput)
    quantity: float


class PositionValue(pa.DataFrameModel):
    position_id: str
    book: str
    market_value: float


class PnLReport(pa.DataFrameModel):
    book: str
    pnl: float


class RiskReport(pa.DataFrameModel):
    book: str
    dv01: float


# A valuer maps (positions, market) -> position-level market value. The instrument
# catalog is captured by make_valuer, so this signature stays clean and injectable.
ValueFn = Callable[["LazyFrame[Positions]", MarketSnapshot], "LazyFrame[PositionValue]"]


# ---------------------------------------------------------------------------
# Valuation atom
# ---------------------------------------------------------------------------
def make_valuer(swaps: LazyFrame[SwapInput]) -> ValueFn:
    """Build a valuer that closes over the instrument catalog. The catalog is
    priced ONCE per market call, then joined to positions and scaled by quantity
    (so N positions on the same swap trigger one pricing pass, not N)."""

    @pa.check_types(lazy=True)
    def value(positions: LazyFrame[Positions], market: MarketSnapshot) -> LazyFrame[PositionValue]:
        priced = price_swap(swaps, market).select("swap_id", "npv")
        result = (
            positions.join(priced, on="swap_id")
            .with_columns(market_value=pl.col("quantity") * pl.col("npv"))
            .select("position_id", "book", "market_value")
        )
        return cast(LazyFrame[PositionValue], result)

    return value


# ---------------------------------------------------------------------------
# Higher-order pipelines (inject `value`)
# ---------------------------------------------------------------------------
@pa.check_types(lazy=True)
def compute_pnl(
    positions: LazyFrame[Positions],
    market_today: MarketSnapshot,
    market_prev: MarketSnapshot,
    *,
    value: ValueFn,
) -> LazyFrame[PnLReport]:
    """Value the same book under two markets, diff, aggregate by book."""
    today = value(positions, market_today).rename({"market_value": "mv_today"})
    prev = value(positions, market_prev).rename({"market_value": "mv_prev"})
    legs = today.join(prev, on=["position_id", "book"]).with_columns(
        pnl=pl.col("mv_today") - pl.col("mv_prev")
    )
    return cast(LazyFrame[PnLReport], legs.group_by("book").agg(pnl=pl.col("pnl").sum()))


@pa.check_types(lazy=True)
def compute_dv01(
    positions: LazyFrame[Positions],
    market: MarketSnapshot,
    bumped: MarketSnapshot,
    *,
    value: ValueFn,
) -> LazyFrame[RiskReport]:
    """Value base + bumped market, diff, aggregate by book. Bucketed DV01 is the
    same shape with a dict of per-tenor bumped markets instead of one."""
    base = value(positions, market).rename({"market_value": "base_mv"})
    up = value(positions, bumped).rename({"market_value": "bumped_mv"})
    legs = base.join(up, on=["position_id", "book"]).with_columns(
        dv01=pl.col("bumped_mv") - pl.col("base_mv")
    )
    return cast(LazyFrame[RiskReport], legs.group_by("book").agg(dv01=pl.col("dv01").sum()))


def bump_curve(market: MarketSnapshot, shift: float) -> MarketSnapshot:
    """Parallel zero-rate shift. The extension point for bucketed bumps: shift only
    the rows matching a tenor bucket instead of the whole curve."""
    return MarketSnapshot(
        as_of=market.as_of,
        curves=market.curves.with_columns(pl.col("zero_rate") + shift),
        fixings=market.fixings,
        projected_indexes=market.projected_indexes,
    )
