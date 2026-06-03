from __future__ import annotations

from datetime import date

import pandera.polars as pa

from schenberg.domain.base import DataFrameModel


class SwapInput(DataFrameModel):
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


class SwapLegInput(DataFrameModel):
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


class SwapOutput(DataFrameModel):
    swap_id: str
    npv: float
    ativo_pv: float
    passivo_pv: float


class LegPricing(DataFrameModel):
    year_fraction: float
    discount_factor: float
    cashflow_amount: float
    signed_cashflow: float
    pv: float
