from __future__ import annotations

from schenberg.domain.base import DataFrameModel


class Position(DataFrameModel):
    position_id: str
    book: str
    instrument_type: str
    instrument_id: str
    quantity: float


class InstrumentValue(DataFrameModel):
    instrument_type: str
    instrument_id: str
    value: float


class PositionValue(DataFrameModel):
    position_id: str
    book: str
    instrument_type: str
    instrument_id: str
    quantity: float
    unit_value: float
    market_value: float
