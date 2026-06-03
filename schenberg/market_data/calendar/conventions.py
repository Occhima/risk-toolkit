from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from schenberg.domain.enums import AccrualConvention

__all__ = ["AccrualConvention", "Calendar"]


@dataclass(frozen=True, slots=True)
class Calendar:
    business_days_per_year: int
    accrual: AccrualConvention
    holidays: frozenset[date]

    @classmethod
    def business_252(
        cls,
        holidays: set[date] | frozenset[date],
        *,
        accrual: AccrualConvention = AccrualConvention.COMPOUND,
    ) -> Calendar:
        return cls(
            business_days_per_year=252,
            accrual=accrual,
            holidays=frozenset(holidays),
        )
