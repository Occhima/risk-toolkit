"""Public boundary contracts. Pandera is used ONLY here, not internally."""

from __future__ import annotations

from datetime import date

import pandera.polars as pa


class SwapInput(pa.DataFrameModel):
    class Config:
        coerce = True  # all-null optional legs infer as Null dtype; coerce to Float64

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


class SwapOutput(pa.DataFrameModel):
    swap_id: str
    npv: float
    ativo_pv: float
    passivo_pv: float


# Shared output contract for every leg family. Drives with_outputs (identity per
# field; override only the renames). Holds exactly what aggregation needs; family
# audit columns (inflation_factor, projected_rate, ...) stay available via stage().
class LegPricing(pa.DataFrameModel):
    year_fraction: float
    discount_factor: float
    cashflow_amount: float
    signed_cashflow: float
    pv: float


# ---------------------------------------------------------------------------
# OPTIONAL: market-feed contracts. These validate the MarketSnapshot frames at
# the data boundary (external feeds), NOT the requirement bindings. They are not
# required for pricing; use them when loading market data, e.g.:
#     CurveTable.validate(curves_df)
# ---------------------------------------------------------------------------
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


class ForwardPricing(pa.DataFrameModel):
    """Reusable pricing output for forward-like instruments."""

    future_value: float
    present_value: float
    value: float


class EnergyForward(pa.DataFrameModel):
    """Block energy forward input.

    ``delivery_periods`` is intentionally left as a runtime list column rather
    than a Pandera field so Polars can explode it without object coercion.
    """

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
