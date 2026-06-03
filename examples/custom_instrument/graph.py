"""The inflation-linked energy forward graph -- a fully custom instrument.

This is "another graph" built node-by-node on the same engine the built-in
instruments use. The payoff is the energy spread, scaled to nominal terms by an
inflation factor, then discounted and converted to the reporting currency:

    real_spread     = forward_price - strike
    inflation_factor = projected_index / base_index
    future_value    = real_spread * inflation_factor
    present_value   = future_value * discount_factor
    value           = present_value * fx_rate

The index-specific part (which calendar date the factor is read at) lives in
``conventions.py`` and arrives as the ``reference_date`` join key, so this graph
stays index-agnostic: one graph prices both IPCA and CPI contracts.
"""

from __future__ import annotations

import polars as pl
from schenberg.core.columns import ColumnSet
from schenberg.core.graph import FormulaGraph
from schenberg.core.market import MarketRequirement
from schenberg.math.expressions import (
    continuous_discount_factor_expr,
    year_fraction_252_expr,
)

inflation_energy_graph = FormulaGraph("inflation_energy_forward")


@inflation_energy_graph.formula(tags=("time",), description="252-day year fraction.")
def year_fraction(payment_days: pl.Expr) -> pl.Expr:
    return year_fraction_252_expr(payment_days)


@inflation_energy_graph.formula(tags=("discounting",))
def discount_factor(zero_rate: pl.Expr, year_fraction: pl.Expr) -> pl.Expr:
    return continuous_discount_factor_expr(zero_rate, year_fraction)


@inflation_energy_graph.formula(tags=("inflation",), description="projected / base index.")
def inflation_factor(projected_index: pl.Expr, base_index: pl.Expr) -> pl.Expr:
    return projected_index / base_index


@inflation_energy_graph.formula(tags=("cashflow",), description="Energy spread in real terms.")
def real_spread(forward_price: pl.Expr, strike: pl.Expr) -> pl.Expr:
    return forward_price - strike


@inflation_energy_graph.formula(tags=("cashflow",), description="Spread scaled to nominal terms.")
def future_value(real_spread: pl.Expr, inflation_factor: pl.Expr) -> pl.Expr:
    return real_spread * inflation_factor


@inflation_energy_graph.formula(tags=("pricing",))
def present_value(future_value: pl.Expr, discount_factor: pl.Expr) -> pl.Expr:
    return future_value * discount_factor


@inflation_energy_graph.formula(tags=("pricing", "fx"))
def value(present_value: pl.Expr, fx_rate: pl.Expr) -> pl.Expr:
    return present_value * fx_rate


inflation_energy_graph.returns(
    "pricing",
    future_value="future_value",
    present_value="present_value",
    value="value",
)

# Market bindings. Each is a left join described by its key bindings; the
# inflation curve joins on (id_indexador, reference_date), so the convention-
# specific date is what selects the right point.
inflation_energy_graph.uses_market(
    MarketRequirement(
        table="inflation_curve",
        on=ColumnSet.from_pairs(("id_indexador", "id_indexador"), ("reference_date", "ref_date")),
        outputs={"projected_index": "projected_index"},
    ),
    MarketRequirement(
        table="inflation_fixings",
        on=ColumnSet.from_pairs(("id_indexador", "id_indexador")),
        outputs={"base_index": "base_index"},
    ),
    MarketRequirement(
        table="di_curve",
        on=ColumnSet.from_pairs(("payment_days", "tenor_days")),
        outputs={"zero_rate": "zero_rate"},
    ),
    MarketRequirement(
        table="fx_rates",
        on=ColumnSet.from_pairs(("currency", "currency")),
        outputs={"fx_rate": "fx_rate"},
    ),
)
