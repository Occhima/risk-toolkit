from __future__ import annotations

from typing import Any

import polars as pl

from schenberg.core import Formula, uses
from schenberg.domain.base import SchenbergDataFrameModel
from schenberg.market_data.requirements import MarketRequirements
from schenberg.math.expressions import (
    continuous_discount_factor_expr,
    year_fraction_252_expr,
)
from schenberg.pricing.instruments.derivatives.forwards.contracts import (
    ForwardContractPricing,
    ForwardPricing,
)
from schenberg.pricing.instruments.derivatives.forwards.market import ForwardMarket


def forward_payoff_expr(forward_price: pl.Expr, strike: pl.Expr) -> pl.Expr:
    """Forward future value before discounting."""
    return forward_price - strike


def present_value_expr(future_value: pl.Expr, discount_factor: pl.Expr) -> pl.Expr:
    """Discount a future value."""
    return future_value * discount_factor


def build_forward_formula(
    *,
    name: str,
    contract: type[SchenbergDataFrameModel] = ForwardContractPricing,
    market: type[MarketRequirements[Any]] = ForwardMarket,
    output: type[SchenbergDataFrameModel] = ForwardPricing,
) -> Formula:
    """Build a reusable typed forward-pricing formula.

    The single implementation of the forward math.  Specialized forwards pass a
    different contract schema and market requirements; the formulas are identical.
    Market terms come from the MarketRequirements type parameter — do not attach
    reads manually inside this builder.
    """
    formula = Formula[contract, market, output](name)

    c = formula.contract
    m = formula.market

    @formula.formula(symbol="T", description="252-business-day year fraction.")
    def year_fraction(payment_days: pl.Expr = uses(c.payment_days)) -> pl.Expr:
        return year_fraction_252_expr(payment_days)

    @formula.formula(symbol="DF", description="Continuous risk-free discount factor.")
    def discount_factor(
        risk_free: pl.Expr = uses(m.risk_free),
        T: pl.Expr = uses(year_fraction),
    ) -> pl.Expr:
        return continuous_discount_factor_expr(risk_free, T)

    @formula.formula(symbol="FV", description="Forward price minus strike.")
    def future_value(
        forward_price: pl.Expr = uses(m.forward_price),
        strike: pl.Expr = uses(c.strike),
    ) -> pl.Expr:
        return forward_payoff_expr(forward_price, strike)

    @formula.formula(symbol="PV", description="Discounted forward value.")
    def present_value(
        fv: pl.Expr = uses(future_value),
        df: pl.Expr = uses(discount_factor),
    ) -> pl.Expr:
        return present_value_expr(fv, df)

    @formula.formula(symbol="V", description="Own-currency present value.")
    def value(pv: pl.Expr = uses(present_value)) -> pl.Expr:
        return pv

    formula.returns(
        "output",
        output,
        future_value=future_value,
        present_value=present_value,
        value=value,
    )
    return formula


forward_formula = build_forward_formula(
    name="forward",
    contract=ForwardContractPricing,
    market=ForwardMarket,
)
