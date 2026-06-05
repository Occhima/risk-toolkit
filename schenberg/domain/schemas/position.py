from __future__ import annotations

from schenberg.domain.base import SchenbergDataFrameModel


class Position(SchenbergDataFrameModel):
    position_id: str
    book: str
    instrument_type: str
    instrument_id: str
    quantity: float
    side: float


class InstrumentPrice(SchenbergDataFrameModel):
    instrument_type: str
    instrument_id: str
    price: float


class PricedPosition(Position):
    price: float
    mtm: float
