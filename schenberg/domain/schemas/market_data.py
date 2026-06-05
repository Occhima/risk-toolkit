from __future__ import annotations

from datetime import date

from schenberg.domain.base import SchenbergDataFrameModel


class DiCurveContract(SchenbergDataFrameModel):
    curve_name: str
    id_indexador: int
    tenor_days: int
    zero_rate: float


class FxRatesContract(SchenbergDataFrameModel):
    currency: str
    fx_rate: float


class EnergyForwardCurveContract(SchenbergDataFrameModel):
    submarket: str
    delivery_period: str
    forward_price: float
    settle_days: int


class FixingContract(SchenbergDataFrameModel):
    id_indexador: int
    fixing_date: date
    fixing_value: float


class VolSurfaceContract(SchenbergDataFrameModel):
    """Implied-volatility quotes on a rectangular (tenor, strike) grid."""

    id_indexador: int
    tenor_days: int
    strike: float
    implied_vol: float
