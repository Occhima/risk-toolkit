from __future__ import annotations

import math

import polars as pl

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def year_fraction_252_expr(payment_days: pl.Expr) -> pl.Expr:
    return payment_days / 252.0


def continuous_discount_factor_expr(rate: pl.Expr, year_fraction: pl.Expr) -> pl.Expr:
    return (-rate * year_fraction).exp()


def norm_pdf_expr(x: pl.Expr) -> pl.Expr:
    """Standard-normal density, vectorized and pure Polars."""
    return (-(x * x) / 2.0).exp() * _INV_SQRT_2PI


def norm_cdf_expr(x: pl.Expr) -> pl.Expr:
    """Standard-normal CDF, Abramowitz & Stegun 26.2.17 — vectorized, pure
    Polars, no map_elements (~7.5e-8 abs error). N(x) = 1 - N(-x)."""
    ax = x.abs()
    t = 1.0 / (1.0 + 0.2316419 * ax)
    poly = t * (
        0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    cdf_pos = 1.0 - norm_pdf_expr(ax) * poly
    return pl.when(x >= 0).then(cdf_pos).otherwise(1.0 - cdf_pos)
