"""The inflation-linked energy forward graph -- a fully custom instrument.

This is "another instrument" built on the same typed engine the built-ins use.
It shows the whole shape of the contract-oriented DSL on a graph that reads
*custom* market tables:

* a ``Contract`` schema (:class:`InflationEnergyLeg`),
* a :class:`MarketRequirements` schema whose fields are ``requires(...)`` over
  hand-rolled :class:`~schenberg.market_data.requirements.Keyed` reads -- you are
  not limited to the built-in market registry,
* pure formulas that only ``uses`` contract and market terms, and an ``Output``.

The payoff is the energy spread, scaled to nominal terms by an inflation factor,
then discounted and converted to the reporting currency::

    real_spread      = forward_price - strike
    inflation_factor = projected_index / base_index
    future_value     = real_spread * inflation_factor
    present_value    = future_value * discount_factor
    value            = present_value * fx_rate

The index-specific part (which calendar date the factor is read at) lives in
``conventions.py`` and arrives as the ``reference_date`` join key, so this graph
stays index-agnostic: one graph prices both IPCA and CPI contracts.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from schenberg.core.graph import Formula, Term, uses
from schenberg.domain.base import SchenbergDataFrameModel
from schenberg.market_data.requirements import Key, Keyed, MarketRequirements, requires
from schenberg.math.expressions import (
    continuous_discount_factor_expr,
    year_fraction_252_expr,
)


class InflationEnergyLeg(SchenbergDataFrameModel):
    """One delivery period of an inflation-linked energy forward."""

    instrument_id: str
    id_indexador: int
    payment_days: int
    forward_price: float
    strike: float
    currency: str
    reference_date: date  # convention-derived inflation-curve join key


class InflationEnergyPricing(SchenbergDataFrameModel):
    future_value: float
    present_value: float
    value: float


def _curve(table: str, value_col: str, *keys: Key) -> Keyed:
    return Keyed(table=table, value_col=value_col, keys=keys)


class InflationEnergyRequirements(MarketRequirements[InflationEnergyLeg]):
    """Custom market reads — the inflation curve joins on the convention-derived
    ``reference_date``, so the index-specific date selects the right point."""

    projected_index: Term[float] = requires(
        _curve(
            "inflation_curve",
            "projected_index",
            Key("indexer", quote_col="id_indexador", default="id_indexador"),
            Key("ref", quote_col="ref_date", default="reference_date"),
        )
    )
    base_index: Term[float] = requires(
        _curve(
            "inflation_fixings",
            "base_index",
            Key("indexer", quote_col="id_indexador", default="id_indexador"),
        )
    )
    zero_rate: Term[float] = requires(
        _curve(
            "di_curve",
            "zero_rate",
            Key("tenor", quote_col="tenor_days", default="payment_days"),
        )
    )
    fx_rate: Term[float] = requires(
        _curve(
            "fx_rates",
            "fx_rate",
            Key("currency", quote_col="currency", default="currency"),
        )
    )


inflation_energy_graph = Formula[
    InflationEnergyLeg,
    InflationEnergyRequirements,
    InflationEnergyPricing,
]("inflation_energy_forward")

c = inflation_energy_graph.contract
m = inflation_energy_graph.market


@inflation_energy_graph.formula(tags=("time",), description="252-day year fraction.")
def year_fraction(d: Term[int] = uses(c.payment_days)) -> pl.Expr:
    return year_fraction_252_expr(d)


@inflation_energy_graph.formula(tags=("discounting",))
def discount_factor(
    r: Term[float] = uses(m.zero_rate), T: Term[float] = uses(year_fraction)
) -> pl.Expr:
    return continuous_discount_factor_expr(r, T)


@inflation_energy_graph.formula(tags=("inflation",), description="projected / base index.")
def inflation_factor(
    projected: Term[float] = uses(m.projected_index), base: Term[float] = uses(m.base_index)
) -> pl.Expr:
    return projected / base


@inflation_energy_graph.formula(tags=("cashflow",), description="Energy spread in real terms.")
def real_spread(
    fwd: Term[float] = uses(c.forward_price), k: Term[float] = uses(c.strike)
) -> pl.Expr:
    return fwd - k


@inflation_energy_graph.formula(tags=("cashflow",), description="Spread scaled to nominal terms.")
def future_value(
    spread: Term[float] = uses(real_spread), factor: Term[float] = uses(inflation_factor)
) -> pl.Expr:
    return spread * factor


@inflation_energy_graph.formula(tags=("pricing",))
def present_value(
    fv: Term[float] = uses(future_value), df: Term[float] = uses(discount_factor)
) -> pl.Expr:
    return fv * df


@inflation_energy_graph.formula(tags=("pricing", "fx"))
def value(pv: Term[float] = uses(present_value), fx: Term[float] = uses(m.fx_rate)) -> pl.Expr:
    return pv * fx


inflation_energy_graph.returns()
