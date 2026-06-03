"""Inflation index conventions: where on the calendar each index is read.

This is the only part of the instrument that is genuinely index-specific. An
inflation-linked energy contract reads its index factor at a *reference date*
derived from the contract's tenor, and the rule differs per index:

- IPCA  (Brazil) -> the first day of the tenor's year      (tenor Jun/2029 -> 2029-01-01)
- CPI   (US)     -> April of the tenor's year              (tenor Jun/2028 -> 2028-04-01)

Adding a new index is a one-line entry in ``CONVENTIONS`` -- no new graph, no new
branch anywhere else. The reference date is built as a single Polars expression
(``reference_date_expr``) so it stays a pure, lazy column.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True, slots=True)
class InflationConvention:
    id_indexador: int
    name: str
    reference_month: int  # month of the tenor's year where the factor is read


# The registry. One row per index; extend it to onboard a new index.
CONVENTIONS: tuple[InflationConvention, ...] = (
    InflationConvention(id_indexador=20, name="IPCA", reference_month=1),  # 1st day of year
    InflationConvention(id_indexador=10, name="CPI", reference_month=4),  # April
)


def reference_date_expr(
    *, tenor_col: str = "tenor_date", index_col: str = "id_indexador"
) -> pl.Expr:
    """Map each row's tenor to the calendar date where its index factor is read,
    dispatching on ``id_indexador`` via the convention registry."""
    year = pl.col(tenor_col).dt.year()
    expr = pl.when(pl.col(index_col) == CONVENTIONS[0].id_indexador).then(
        pl.date(year, CONVENTIONS[0].reference_month, 1)
    )
    for conv in CONVENTIONS[1:]:
        expr = expr.when(pl.col(index_col) == conv.id_indexador).then(
            pl.date(year, conv.reference_month, 1)
        )
    return expr.otherwise(None).alias("reference_date")


def add_reference_date(legs: pl.LazyFrame) -> pl.LazyFrame:
    """Normalization step: attach the index-specific ``reference_date`` column
    that the pricing graph later joins the inflation curve on. Runs *before* the
    graph because join keys must exist before the market attach."""
    return legs.with_columns(reference_date_expr())
