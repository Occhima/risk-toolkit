"""Public boundary contracts. Pandera is used ONLY here, not internally."""

from __future__ import annotations

from datetime import date

import pandera.polars as pa

from schenberg.domain.base import DataFrameModel
from schenberg.domain.schemas.forward import EnergyForwardLeg, ForwardPricing, ForwardTrade
from schenberg.domain.schemas.market_data import (
    DiCurveContract,
    EnergyForwardCurveContract,
    FixingContract,
    FxRatesContract,
)
from schenberg.domain.schemas.position import (
    InstrumentPrice,
    Position,
    PricedPosition,
)
from schenberg.domain.schemas.swap import LegPricing, SwapInput, SwapLegInput, SwapOutput


class CurveTable(DataFrameModel):
    id_indexador: int
    tenor_days: int
    zero_rate: float
    forward_rate: float = pa.Field(nullable=True)
    discount_factor: float = pa.Field(nullable=True)


class FixingTable(DataFrameModel):
    id_indexador: int
    fixing_date: date
    fixing_value: float


class ProjectedTable(DataFrameModel):
    id_indexador: int
    tenor_days: int
    projected_index: float


class EnergyForward(DataFrameModel):
    """Legacy block energy forward input retained for compatibility only."""

    contract_id: str
    submarket: str
    buy_sell: str
    id_indexador: int
    quantity: float
    strike: float
    currency: str


class EnergyForwardOutput(DataFrameModel):
    instrument_id: str
    mtm_local: float
    mtm: float


__all__ = [
    "DataFrameModel",
    "DiCurveContract",
    "EnergyForward",
    "EnergyForwardCurveContract",
    "EnergyForwardLeg",
    "EnergyForwardOutput",
    "FixingContract",
    "FixingTable",
    "ForwardPricing",
    "ForwardTrade",
    "FxRatesContract",
    "InstrumentPrice",
    "LegPricing",
    "Position",
    "PricedPosition",
    "ProjectedTable",
    "SwapInput",
    "SwapLegInput",
    "SwapOutput",
]
