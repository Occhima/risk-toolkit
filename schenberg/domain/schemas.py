"""Public boundary contracts. Pandera is used ONLY here, not internally."""

from __future__ import annotations

from datetime import date

import pandera.polars as pa


class SwapInput(pa.DataFrameModel):
    class Config:
        coerce = True

    swap_id: str
    notional: float
    id_indexador_ativo: int
    id_indexador_passivo: int
    indexador_kind_ativo: str
    indexador_kind_passivo: str
    payment_days: int
    accrual: float
    base_date: date
    fixed_rate_ativo: float = pa.Field(nullable=True)
    fixed_rate_passivo: float = pa.Field(nullable=True)
    real_coupon_ativo: float = pa.Field(nullable=True)
    real_coupon_passivo: float = pa.Field(nullable=True)


class SwapLegInput(pa.DataFrameModel):
    class Config:
        coerce = True

    swap_id: str
    leg_id: str
    leg_kind: str
    pay_receive: str
    notional: float
    id_indexador: int
    payment_days: int
    accrual: float
    base_date: date
    fixed_rate: float = pa.Field(nullable=True)
    real_coupon: float = pa.Field(nullable=True)
    cashflow_amount: float = pa.Field(nullable=True)


class SwapOutput(pa.DataFrameModel):
    swap_id: str
    npv: float
    ativo_pv: float
    passivo_pv: float


class LegPricing(pa.DataFrameModel):
    year_fraction: float
    discount_factor: float
    cashflow_amount: float
    signed_cashflow: float
    pv: float


class CurveTable(pa.DataFrameModel):
    id_indexador: int
    tenor_days: int
    zero_rate: float
    forward_rate: float = pa.Field(nullable=True)
    discount_factor: float = pa.Field(nullable=True)


class FixingTable(pa.DataFrameModel):
    id_indexador: int
    fixing_date: date
    fixing_value: float


class ProjectedTable(pa.DataFrameModel):
    id_indexador: int
    tenor_days: int
    projected_index: float


class ForwardTrade(pa.DataFrameModel):
    class Config:
        coerce = True

    trade_id: str
    instrument_type: str
    forward_family: str
    settlement_type: str
    currency: str
    id_indexador: int
    payment_days: int
    future_value: float = pa.Field(nullable=True)


class EnergyForwardLeg(ForwardTrade):
    contract_id: str
    submarket: str
    delivery_period: str
    buy_sell: str
    quantity: float
    strike: float


class ForwardPricing(pa.DataFrameModel):
    """Reusable pricing output for forward-like instruments."""

    future_value: float
    present_value: float
    value: float


class EnergyForward(pa.DataFrameModel):
    """Legacy block energy forward input retained for compatibility only."""

    contract_id: str
    submarket: str
    buy_sell: str
    id_indexador: int
    quantity: float
    strike: float
    currency: str


class EnergyForwardOutput(pa.DataFrameModel):
    contract_id: str
    mtm_local: float
    mtm: float
