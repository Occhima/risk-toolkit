"""Pure Polars-expression helpers for deriving market-data join-key dates.

Usage pattern::

    from schenberg.market_data.date_rules import start_of_tenor_year

    prepared = legs.with_columns(
        start_of_tenor_year(tenor_col="tenor_date", output_col="pca_fixing_date")
    )

The resulting column is used as a join key in a MarketRequirement; it is never
read by the graph formulas themselves.
"""

from __future__ import annotations

import polars as pl


def start_of_tenor_year(
    *,
    tenor_col: str = "tenor_date",
    output_col: str = "fixing_date",
) -> pl.Expr:
    """Map each row's tenor date to January 1st of the same year."""
    year = pl.col(tenor_col).dt.year()
    return pl.date(year, 1, 1).alias(output_col)


def first_day_of_tenor_month(
    *,
    tenor_col: str = "tenor_date",
    output_col: str = "fixing_date",
) -> pl.Expr:
    """Map each row's tenor date to the first day of the same month."""
    tenor = pl.col(tenor_col)
    return pl.date(tenor.dt.year(), tenor.dt.month(), 1).alias(output_col)


_MONTH_MIN = 1
_MONTH_MAX = 12


def constant_month_of_tenor_year(
    *,
    month: int,
    tenor_col: str = "tenor_date",
    output_col: str = "fixing_date",
) -> pl.Expr:
    """Map each row's tenor date to the first day of ``month`` in the same year."""
    if not _MONTH_MIN <= month <= _MONTH_MAX:
        raise ValueError(f"month must be between 1 and 12, got {month}")
    year = pl.col(tenor_col).dt.year()
    return pl.date(year, month, 1).alias(output_col)


def copy_date(
    *,
    source_col: str,
    output_col: str = "fixing_date",
) -> pl.Expr:
    """Alias an existing date column under a new name for use as a join key."""
    return pl.col(source_col).alias(output_col)


def with_date_rule(lf: pl.LazyFrame, expr: pl.Expr) -> pl.LazyFrame:
    """Convenience wrapper: apply a date-rule expression to a LazyFrame."""
    return lf.with_columns(expr)
