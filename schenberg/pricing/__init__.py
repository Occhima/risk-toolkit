"""Pricing namespace.

Schenberg keeps pricing helpers lazy: functions return ``polars.LazyFrame`` and
never collect. Instrument-specific pricers can still be declared by applications;
this namespace exposes the small generic forward helper used by plugin examples.
"""

from __future__ import annotations

import polars as pl

from schenberg.core.expr import exp
from schenberg.core.graph import FormulaGraph
from schenberg.domain.base import SchenbergDataFrameModel
from schenberg.market_data.roles import With, bind, market_role
from schenberg.market_data.snapshot import MarketSnapshot

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


# ``With[role]`` builds a pandera ``DataFrameModel`` mixin dynamically; ty cannot
# compute an MRO across pandera's metaclass and flags the base. The pattern is the
# documented way to declare resolved market columns on an input schema.
class ForwardInput(With[ForwardRate], With[RiskFreeRate], SchenbergDataFrameModel):  # ty: ignore[unsupported-base]
    instrument_id: str
    indexer: str
    currency: str
    strike: float
    payment_days: int


forward_formula = FormulaGraph("forward", input=ForwardInput)


@forward_formula.formula(symbol="T")
def year_fraction(payment_days):
    return payment_days / 252.0


@forward_formula.formula(symbol="DF")
def discount_factor(risk_free_rate, year_fraction):
    return exp(-risk_free_rate * year_fraction)


@forward_formula.formula(symbol="FV")
def future_value(forward_rate, strike):
    return forward_rate - strike


@forward_formula.formula(symbol="PV")
def present_value(future_value, discount_factor):
    return future_value * discount_factor


@forward_formula.formula(symbol="Delta")
def delta(discount_factor):
    return discount_factor


forward_formula.returns(
    "output",
    instrument_id="instrument_id",
    future_value="future_value",
    present_value="present_value",
    value="present_value",
    delta="delta",
    currency="currency",
)


def price_forward(trades: pl.LazyFrame, market: MarketSnapshot) -> pl.LazyFrame:
    """Price generic forward instruments against a market snapshot lazily."""
    enriched = bind(trades, market, ForwardInput)
    return forward_formula.plan(enriched, view="output")


__all__ = ["ForwardInput", "forward_formula", "price_forward"]
