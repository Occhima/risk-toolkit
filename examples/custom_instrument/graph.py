"""The inflation-linked energy forward graph -- a fully custom instrument.

This is "another graph" built on the same engine the built-in instruments use,
in the canonical Term DSL: inputs come from :data:`InflationEnergyLeg` via
``g.input``, the market reads are declared as terms with ``g.market``, and every
formula names its dependencies with ``uses``. The payoff is the energy spread,
scaled to nominal terms by an inflation factor, then discounted and converted to
the reporting currency::

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

import polars as pl
from schenberg.core.graph import FormulaGraph, uses
from schenberg.domain.base import DataFrameModel
from schenberg.market_data.specs import JoinSpec
from schenberg.math.expressions import (
    continuous_discount_factor_expr,
    year_fraction_252_expr,
)


class InflationEnergyLeg(DataFrameModel):
    """One delivery period of an inflation-linked energy forward."""

    instrument_id: str
    id_indexador: int
    payment_days: int
    forward_price: float
    strike: float
    currency: str
    reference_date: object  # the convention-derived inflation-curve join key


inflation_energy_graph = FormulaGraph("inflation_energy_forward", input=InflationEnergyLeg)
t = inflation_energy_graph.input

# Market reads as graph terms. Each is a keyed left join; the inflation curve
# joins on (id_indexador, reference_date -> ref_date), so the convention-specific
# date is what selects the right point.
m = inflation_energy_graph.market(
    projected_index=JoinSpec("inflation_curve").read(
        "projected_index", ("id_indexador", "id_indexador"), ("reference_date", "ref_date")
    ),
    base_index=JoinSpec("inflation_fixings").read("base_index", ("id_indexador", "id_indexador")),
    zero_rate=JoinSpec("di_curve").read("zero_rate", ("payment_days", "tenor_days")),
    fx_rate=JoinSpec("fx_rates").read("fx_rate", ("currency", "currency")),
)


@inflation_energy_graph.formula(tags=("time",), description="252-day year fraction.")
def year_fraction(d: pl.Expr = uses(t.payment_days)) -> pl.Expr:
    return year_fraction_252_expr(d)


@inflation_energy_graph.formula(tags=("discounting",))
def discount_factor(r: pl.Expr = uses(m.zero_rate), T: pl.Expr = uses(year_fraction)) -> pl.Expr:
    return continuous_discount_factor_expr(r, T)


@inflation_energy_graph.formula(tags=("inflation",), description="projected / base index.")
def inflation_factor(
    projected: pl.Expr = uses(m.projected_index), base: pl.Expr = uses(m.base_index)
) -> pl.Expr:
    return projected / base


@inflation_energy_graph.formula(tags=("cashflow",), description="Energy spread in real terms.")
def real_spread(fwd: pl.Expr = uses(t.forward_price), k: pl.Expr = uses(t.strike)) -> pl.Expr:
    return fwd - k


@inflation_energy_graph.formula(tags=("cashflow",), description="Spread scaled to nominal terms.")
def future_value(
    spread: pl.Expr = uses(real_spread), factor: pl.Expr = uses(inflation_factor)
) -> pl.Expr:
    return spread * factor


@inflation_energy_graph.formula(tags=("pricing",))
def present_value(fv: pl.Expr = uses(future_value), df: pl.Expr = uses(discount_factor)) -> pl.Expr:
    return fv * df


@inflation_energy_graph.formula(tags=("pricing", "fx"))
def value(pv: pl.Expr = uses(present_value), fx: pl.Expr = uses(m.fx_rate)) -> pl.Expr:
    return pv * fx


inflation_energy_graph.returns(
    "pricing",
    future_value=future_value,
    present_value=present_value,
    value=value,
)
