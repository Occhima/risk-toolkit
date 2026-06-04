"""Public boundary contracts. Pandera is used ONLY here, not internally."""

from __future__ import annotations

from schenberg.domain.base import DataFrameModel
from schenberg.domain.schemas.forward import EnergyForwardLeg, ForwardPricing, ForwardTrade
from schenberg.domain.schemas.market_data import (
    DiCurveContract,
    EnergyForwardCurveContract,
    FixingContract,
    FxRatesContract,
)
from schenberg.domain.schemas.position import InstrumentPrice, Position, PricedPosition
from schenberg.domain.schemas.structure import StructureLeg
from schenberg.domain.schemas.swap import (
    LegPricing,
    SwapLegInput,
    SwapLegStage,
    SwapOutput,
)

__all__ = [
    "DataFrameModel",
    "DiCurveContract",
    "EnergyForwardCurveContract",
    "EnergyForwardLeg",
    "FixingContract",
    "ForwardPricing",
    "ForwardTrade",
    "FxRatesContract",
    "InstrumentPrice",
    "LegPricing",
    "Position",
    "PricedPosition",
    "StructureLeg",
    "SwapLegInput",
    "SwapLegStage",
    "SwapOutput",
]
