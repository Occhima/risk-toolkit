"""Public boundary contracts. Pandera is used ONLY here, not internally."""

from __future__ import annotations

from schenberg.domain.base import SchenbergDataFrameModel
from schenberg.domain.schemas.forward import EnergyForwardLeg, ForwardPricing, ForwardTrade
from schenberg.domain.schemas.market_data import (
    DiCurveContract,
    EnergyForwardCurveContract,
    FixingContract,
    FxRatesContract,
)
from schenberg.domain.schemas.position import (
    BookContract,
    InstrumentPnlExplain,
    InstrumentRisk,
    InstrumentValue,
    Position,
    PositionPnlExplain,
    PositionRisk,
    PositionValue,
    ReportingFx,
)
from schenberg.domain.schemas.structure import StructureLeg
from schenberg.domain.schemas.swap import (
    LegPricing,
    SwapLegInput,
    SwapLegStage,
    SwapOutput,
)

__all__ = [
    "SchenbergDataFrameModel",
    "DiCurveContract",
    "EnergyForwardCurveContract",
    "EnergyForwardLeg",
    "FixingContract",
    "ForwardPricing",
    "ForwardTrade",
    "FxRatesContract",
    "BookContract",
    "InstrumentPnlExplain",
    "InstrumentRisk",
    "InstrumentValue",
    "LegPricing",
    "Position",
    "PositionPnlExplain",
    "PositionRisk",
    "PositionValue",
    "ReportingFx",
    "StructureLeg",
    "SwapLegInput",
    "SwapLegStage",
    "SwapOutput",
]
