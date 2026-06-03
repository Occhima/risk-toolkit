from __future__ import annotations

from schenberg.domain.base import DataFrameModel


class Position(DataFrameModel):
    position_id: str
    book: str
    instrument_type: str
    instrument_id: str
    quantity: float


class InstrumentPrice(DataFrameModel):
    instrument_type: str
    instrument_id: str
    price: float


class PricedPosition(Position):
    price: float
    mtm: float
