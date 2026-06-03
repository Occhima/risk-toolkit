from __future__ import annotations

from schenberg.domain.base import DataFrameModel


class ForwardTrade(DataFrameModel):
    instrument_id: str
    instrument_type: str
    forward_family: str
    settlement_type: str
    currency: str
    id_indexador: int
    # Optional: generic forwards carry an explicit settlement horizon, while
    # energy forwards derive it from the delivery period's ANBIMA fixing date.
    payment_days: int | None


class EnergyForwardLeg(ForwardTrade):
    submarket: str
    delivery_period: str
    strike: float


class ForwardPricing(DataFrameModel):
    """Reusable pricing output for forward-like instruments."""

    future_value: float
    present_value: float
    value: float
