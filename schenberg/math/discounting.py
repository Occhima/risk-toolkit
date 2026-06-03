from __future__ import annotations

from math import exp


def year_fraction_252(days: int | float) -> float:
    return days / 252.0


def compound_factor(rate: float, year_fraction: float) -> float:
    return (1.0 + rate) ** year_fraction


def compound_discount_factor(rate: float, year_fraction: float) -> float:
    return 1.0 / compound_factor(rate, year_fraction)


def continuous_discount_factor(rate: float, year_fraction: float) -> float:
    return exp(-rate * year_fraction)
