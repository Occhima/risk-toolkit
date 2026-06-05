from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from schenberg.domain.enums import AccrualConvention

__all__ = ["AccrualConvention", "Calendar"]


@dataclass(frozen=True, slots=True)
class Calendar:
    business_days_per_year: int
    accrual: AccrualConvention
    holidays: frozenset[date]
    name: str = "BUS/252"

    @property
    def base_days(self) -> int:
        return self.business_days_per_year

    def business_days_between_expr(self, start_col: str, end_col: str) -> pl.Expr:
        return pl.business_day_count(start_col, end_col, holidays=list(self.holidays))

    @classmethod
    def business_252(
        cls,
        holidays: set[date] | frozenset[date],
        *,
        accrual: AccrualConvention = AccrualConvention.COMPOUND,
        name: str = "BUS/252",
    ) -> Calendar:
        return cls(
            business_days_per_year=252,
            accrual=accrual,
            holidays=frozenset(holidays),
            name=name,
        )
