from __future__ import annotations

from datetime import date

from schenberg.domain.base import DataFrameModel


class ForwardTrade(DataFrameModel):
    instrument_id: str
    instrument_type: str
    forward_family: str
    settlement_type: str
    currency: str
    id_indexador: int
    payment_days: int


class GenericForwardLeg(ForwardTrade):
    """A generic forward leg: a forward price and a strike, valued and discounted."""

    forward_price: float
    strike: float


class EnergyForwardLeg(ForwardTrade):
    submarket: str
    delivery_period: str
    strike: float
    # Settlement/fixing date — part of the input contract. Callers that don't
    # already have it can build it with energy.with_fixing_date (6th ANBIMA
    # business day of the month following delivery).
    fixing_date: date


class ForwardPricing(DataFrameModel):
    """Reusable pricing output for forward-like instruments."""

    future_value: float
    present_value: float
    value: float
