"""Forward and energy-forward public pricers.

All pricers return lazy frames and never call ``.collect()``. Market data is
resolved via :func:`~schenberg.market_data.roles.bind` before the formula graph
runs, so the graph itself is a pure function of its input columns.

Public surface
--------------
price_forward               generic forward (forward_rate from "curves")
forward_instrument_value    same, shaped as InstrumentValue for the position layer
price_energy_forward        energy forward (forward_price from "energy_forward_curve")
ForwardContractPricing      typed input schema (raw + market columns)
EnergyForwardPricing        typed input schema for energy forwards
forward_formula             the underlying FormulaGraph (for inspection/introspection)
energy_forward_formula      same for energy forwards
"""

from __future__ import annotations

import polars as pl

from schenberg.core.expr import exp
from schenberg.core.graph import FormulaGraph
from schenberg.market_data.roles import With, bind, market_role
from schenberg.market_data.snapshot import MarketSnapshot

# ── market roles — generic forward ───────────────────────────────────────────
# Both keys read from a "curves" source keyed by (id_indexador, tenor_days).
# The trade frame uses "indexer" and "payment_days" as the left-join keys.

_ForwardRate = (
    market_role("forward_rate")
    .read("curves", "forward_rate")
    .by(indexer="id_indexador", payment_days="tenor_days")
)

_RiskFreeRate = (
    market_role("risk_free_rate")
    .read("curves", "risk_free_rate")
    .by(indexer="id_indexador", payment_days="tenor_days")
)

# ── market roles — energy forward ─────────────────────────────────────────────
# Energy forward prices live on a separate "energy_forward_curve" source keyed
# by (submarket, delivery_period). The discount rate still comes from "curves".

_EnergyForwardPrice = (
    market_role("forward_rate")
    .read("energy_forward_curve", "forward_price")
    .by(submarket="submarket", delivery_period="delivery_period")
)

_EnergyRiskFreeRate = (
    market_role("risk_free_rate")
    .read("curves", "risk_free_rate")
    .by(indexer="id_indexador", payment_days="tenor_days")
)

# ── input schemas ─────────────────────────────────────────────────────────────


class ForwardContractPricing(With[_ForwardRate], With[_RiskFreeRate]):
    """Typed input for generic forwards.

    The caller supplies the raw contract columns; :func:`bind` joins
    ``forward_rate`` and ``risk_free_rate`` from the market snapshot.
    Inherits from ``SchenbergDataFrameModel`` via the ``With[...]`` mixins.
    """

    instrument_id: str
    indexer: str
    currency: str
    strike: float
    payment_days: int
    # resolved by bind():
    # forward_rate: float
    # risk_free_rate: float


class EnergyForwardPricing(With[_EnergyForwardPrice], With[_EnergyRiskFreeRate]):
    """Typed input for energy forwards.

    Forward prices come from "energy_forward_curve" keyed by
    ``(submarket, delivery_period)``; discounting uses "curves".
    """

    instrument_id: str
    indexer: str
    currency: str
    strike: float
    payment_days: int
    submarket: str
    delivery_period: str
    # resolved by bind():
    # forward_rate: float  (from energy_forward_curve.forward_price)
    # risk_free_rate: float


# ── shared formula graph ──────────────────────────────────────────────────────
# Both pricers share the same formula: the market roles bring forward_rate and
# risk_free_rate into the frame under those exact names, so one graph serves
# both. We declare two graph instances so each can carry its own schema for
# introspection, but they encode the same math.


def _build_forward_graph(name: str, schema: type) -> FormulaGraph:
    g = FormulaGraph(name, input=schema)
    c = g.input
    T = g.let("year_fraction", c.payment_days / 252.0, symbol="T")
    DF = g.let("discount_factor", exp(-c.risk_free_rate * T), symbol="DF")
    FV = g.let("future_value", c.forward_rate - c.strike, symbol="FV")
    PV = g.let("present_value", FV * DF, symbol="PV")
    g.let("value", PV, symbol="V")
    g.returns("output", future_value="future_value", present_value="present_value", value="value")
    return g


# Public formula objects — inspectable via .explain(), .to_mermaid(), .info()

forward_formula: FormulaGraph = _build_forward_graph("forward_pricing", ForwardContractPricing)

energy_forward_formula: FormulaGraph = _build_forward_graph(
    "energy_forward_pricing", EnergyForwardPricing
)


# ── public pricers ────────────────────────────────────────────────────────────


def price_forward(
    trades: pl.LazyFrame | pl.DataFrame,
    market: MarketSnapshot,
) -> pl.LazyFrame:
    """Price generic forwards against a market snapshot.

    Joins ``forward_rate`` and ``risk_free_rate`` from the snapshot's ``curves``
    source (keyed by ``(id_indexador, tenor_days)``), then computes
    ``future_value``, ``present_value`` and ``value`` in the instrument's own
    currency. Returns the enriched trade frame with those three columns appended.
    Stays lazy.

    Parameters
    ----------
    trades:
        LazyFrame with at least ``instrument_id``, ``indexer``, ``currency``,
        ``strike``, ``payment_days``. Extra columns (e.g. ``tenor``) pass
        through bind and are dropped after schema projection.
    market:
        Snapshot whose ``curves`` source has ``id_indexador``, ``tenor_days``,
        ``forward_rate``, and ``risk_free_rate``.
    """
    enriched = bind(trades, market, ForwardContractPricing)
    return forward_formula.plan(enriched, view="output")


def forward_instrument_value(
    trades: pl.LazyFrame | pl.DataFrame,
    market: MarketSnapshot,
    *,
    instrument_type: str = "FORWARD",
) -> pl.LazyFrame:
    """Price forwards and return an ``InstrumentValue`` frame for the position layer."""
    priced = price_forward(trades, market)
    return priced.select(
        instrument_type=pl.lit(instrument_type),
        instrument_id=pl.col("instrument_id"),
        value=pl.col("value"),
        currency=pl.col("currency"),
    )


def price_energy_forward(
    trades: pl.LazyFrame | pl.DataFrame,
    market: MarketSnapshot,
) -> pl.LazyFrame:
    """Price energy forwards against a market snapshot.

    Forward prices are read from the snapshot's ``energy_forward_curve`` source
    (keyed by ``(submarket, delivery_period)``); discounting uses ``curves``.
    Returns the enriched trade frame with ``future_value``, ``present_value``,
    and ``value`` appended. Stays lazy.
    """
    enriched = bind(trades, market, EnergyForwardPricing)
    return energy_forward_formula.plan(enriched, view="output")
