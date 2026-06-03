from __future__ import annotations

from schenberg.domain.base import DataFrameModel


class StructureLeg(DataFrameModel):
    """One component leg of a structured product."""

    structure_id: str
    leg_id: str
    component_instrument_type: str
    component_instrument_id: str
    quantity: float
    side: float
