from __future__ import annotations

import polars as pl


def year_fraction_252_expr(payment_days: pl.Expr) -> pl.Expr:
    return payment_days / 252.0


def continuous_discount_factor_expr(rate: pl.Expr, year_fraction: pl.Expr) -> pl.Expr:
    return (-rate * year_fraction).exp()
