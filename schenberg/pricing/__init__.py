"""Pricing namespace.

Schenberg keeps pricing helpers lazy: functions return ``polars.LazyFrame`` and
never collect. Instrument-specific pricers can still be declared by applications;
this namespace exposes the small generic forward helper used by plugin examples.
"""

from __future__ import annotations

import sys
import types

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


class ForwardInput(With[ForwardRate], With[RiskFreeRate], SchenbergDataFrameModel):
    instrument_id: str
    indexer: str
    currency: str
    strike: float
    payment_days: int


forward_formula = FormulaGraph("forward", input=ForwardInput)


@forward_formula.formula(symbol="T")
def year_fraction(c):
    return c.payment_days / 252.0


@forward_formula.formula(symbol="DF")
def discount_factor(c, year_fraction):
    return exp(-c.risk_free_rate * year_fraction)


@forward_formula.formula(symbol="FV")
def future_value(c):
    return c.forward_rate - c.strike


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


_compat = types.ModuleType(__name__ + ".api")
_compat.ForwardInput = ForwardInput
_compat.forward_formula = forward_formula
_compat.price_forward = price_forward
sys.modules[__name__ + ".api"] = _compat

__all__ = ["ForwardInput", "forward_formula", "price_forward"]
