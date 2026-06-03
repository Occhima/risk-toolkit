from __future__ import annotations

from datetime import date

from schenberg.domain.base import DataFrameModel


class DiCurveContract(DataFrameModel):
    curve_name: str
    id_indexador: int
    tenor_days: int
    zero_rate: float


class FxRatesContract(DataFrameModel):
    currency: str
    fx_rate: float


class EnergyForwardCurveContract(DataFrameModel):
    submarket: str
    delivery_period: str
    forward_price: float


class FixingContract(DataFrameModel):
    id_indexador: int
    fixing_date: date
    fixing_value: float


class VolSurfaceContract(DataFrameModel):
    """Implied-volatility quotes on a rectangular (tenor, strike) grid."""

    id_indexador: int
    tenor_days: int
    strike: float
    implied_vol: float


class CarryCurveContract(DataFrameModel):
    """Cost-of-carry ``b`` by underlying and tenor (the generalized BSM knob)."""

    id_indexador: int
    tenor_days: int
    cost_of_carry: float


class DividendCurveContract(DataFrameModel):
    """Continuous dividend yield ``q`` by underlying and tenor (Merton ``b = r - q``)."""

    id_indexador: int
    tenor_days: int
    div_yield: float
