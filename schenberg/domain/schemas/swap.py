from __future__ import annotations

from datetime import date

import pandera.polars as pa

from schenberg.domain.base import DataFrameModel


class SwapLegInput(DataFrameModel):
    """One swap leg as a normalized row.

    ``leg_weight`` carries the *position direction* (+1 receive / -1 pay, or a
    notional sign): it belongs to the structure layer, never to the pure pricing
    graph. ``leg_role`` is an optional classification (``"ativo"``/``"passivo"``,
    ``"fixed"``/``"floating"``) used only for structure-level reporting splits.
    """

    swap_id: str
    leg_id: str
    leg_kind: str
    leg_role: str = pa.Field(nullable=True)
    leg_weight: float
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
    """Pure component pricing for one leg — no direction, no sign.

    ``pv = cashflow_amount * discount_factor``. The position weight is applied
    later, at the structure layer (see :class:`SwapLegStage`).
    """

    year_fraction: float
    discount_factor: float
    cashflow_amount: float
    pv: float


class SwapLegStage(DataFrameModel):
    """Structure-stage debug view: pure component pricing plus exposure.

    ``weighted_pv = pv * leg_weight`` is the leg's signed contribution to its
    swap. This schema documents the columns :meth:`Structure.stage` exposes; it is
    a debugging contract, not a hot-path one.
    """

    swap_id: str
    leg_id: str
    leg_role: str = pa.Field(nullable=True)
    leg_weight: float
    cashflow_amount: float
    discount_factor: float
    pv: float
    weighted_pv: float
