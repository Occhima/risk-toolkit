from __future__ import annotations

from schenberg.domain.base import DataFrameModel


class ForwardTrade(DataFrameModel):
    instrument_id: str
    instrument_type: str
    forward_family: str
    settlement_type: str
    currency: str
    id_indexador: int
    payment_days: int


class EnergyForwardLeg(ForwardTrade):
    submarket: str
    delivery_period: str
    buy_sell: str
    strike: float


class ForwardPricing(DataFrameModel):
    """Reusable pricing output for forward-like instruments."""

    future_value: float
    present_value: float
    value: float
