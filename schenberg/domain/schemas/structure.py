from __future__ import annotations

from schenberg.domain.base import SchenbergDataFrameModel


class StructureLeg(SchenbergDataFrameModel):
    """One already-priceable leg of a structured product."""

    structure_id: str
    leg_id: str
    instrument_type: str
    instrument_id: str
    quantity: float
    weight: float
