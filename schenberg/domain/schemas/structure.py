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


class StructureLegValue(SchenbergDataFrameModel):
    """Leg-level debug view after priced instruments have been joined."""

    structure_id: str
    leg_id: str
    instrument_type: str
    instrument_id: str
    quantity: float
    weight: float
    value: float
    leg_value: float


class StructureValue(SchenbergDataFrameModel):
    """Aggregated value of a structure."""

    structure_id: str
    value: float
