"""Boundary contracts for the position-computation layer.

The pricing layer answers *what is one unit of this instrument worth?* and emits
an :class:`InstrumentValue` (or a PnL decomposition, :class:`InstrumentPnlExplain`)
— a **pure** value, with no position direction in it. The position layer answers
*how much do I hold, and what is that worth in my book's terms?*: it lifts a pure
instrument value onto a :class:`Position` (one row per economic target) and a
book/reporting :class:`BookContract` / :class:`ReportingFx` context, producing a
:class:`PositionValue` whose columns are *measures* (exposure, notional, mtm,
reported mtm).

``side`` / ``quantity`` live on :class:`Position` and enter only the position
layer's measures — never a pricing graph.
"""

from __future__ import annotations

import pandera.polars as pa

from schenberg.domain.base import SchenbergDataFrameModel

# ---- pricing-layer outputs (pure: no side, no position) ----------------------


class InstrumentValue(SchenbergDataFrameModel):
    """Pure instrument value/price — what a pricer returns.

    One row per priced instrument. ``value`` is the value of *one unit*; it
    carries no position direction. ``currency`` is the value's denomination (a
    property of the instrument, not of the position), used to convert into a
    book's reporting currency in the position layer.
    """

    instrument_type: str
    instrument_id: str
    value: float
    currency: str = pa.Field(nullable=True)


class InstrumentPnlExplain(SchenbergDataFrameModel):
    """A pure per-instrument PnL decomposition (one unit), additive by component.

    ``total_value_pnl`` equals the sum of the attribution components. Each column
    is a pure value change — no position direction.
    """

    instrument_type: str
    instrument_id: str
    currency: str = pa.Field(nullable=True)
    roll_value_pnl: float
    curve_value_pnl: float
    fx_value_pnl: float
    fixing_value_pnl: float
    residual_value_pnl: float
    total_value_pnl: float


class InstrumentRisk(SchenbergDataFrameModel):
    """Pure per-instrument *risk factors* — sensitivities of one unit's value.

    The same shape as :class:`InstrumentValue`, but a vector of sensitivities
    instead of a single value (the closed-form Black-Scholes-Merton Greeks; see
    :class:`~schenberg.domain.schemas.option.OptionGreeks`). Like every pricing
    output it is **pure**: no ``side``, no position. The position layer lifts each
    factor by exposure exactly as it lifts ``value`` into ``mtm``.
    """

    instrument_type: str
    instrument_id: str
    currency: str = pa.Field(nullable=True)
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


# ---- position / book context -------------------------------------------------


class Position(SchenbergDataFrameModel):
    """The economic target: how much of an instrument is held, and in which book.

    ``book`` is a *key* only — desk, legal entity and reporting currency live on
    :class:`BookContract`, joined once, never duplicated on every position row.
    ``side`` and ``quantity`` are position direction/size: they belong here and
    feed the position layer's measures, never a pricing graph.
    """

    position_id: str
    book: str
    instrument_type: str
    instrument_id: str
    quantity: float
    side: float
    unit_notional: float = pa.Field(nullable=True)


class BookContract(SchenbergDataFrameModel):
    """Per-book reference data, joined on ``book``. One row per book."""

    book: str
    desk: str = pa.Field(nullable=True)
    legal_entity: str = pa.Field(nullable=True)
    reporting_currency: str


class ReportingFx(SchenbergDataFrameModel):
    """FX conversion table into a book's reporting currency.

    Keyed by the instrument ``currency`` and the book ``reporting_currency``;
    ``book_fx`` converts a value in ``currency`` into ``reporting_currency``
    (``reported = value / book_fx`` with the units used here).
    """

    currency: str
    reporting_currency: str
    book_fx: float


# ---- position-layer outputs (measures) ---------------------------------------


class PositionValue(SchenbergDataFrameModel):
    """A position valued: the position-layer measures, one row per position."""

    position_id: str
    book: str
    exposure: float
    position_notional: float = pa.Field(nullable=True)
    mtm: float
    reported_mtm: float


class PositionPnlExplain(SchenbergDataFrameModel):
    """Position-level PnL explain: each instrument component lifted to the book."""

    position_id: str
    book: str
    roll_mtm_pnl: float
    curve_mtm_pnl: float
    fx_mtm_pnl: float
    fixing_mtm_pnl: float
    residual_mtm_pnl: float
    total_mtm_pnl: float


class PositionRisk(SchenbergDataFrameModel):
    """Position-level risk factors: each instrument sensitivity scaled by exposure,
    one row per position."""

    position_id: str
    book: str
    position_delta: float
    position_gamma: float
    position_vega: float
    position_theta: float
    position_rho: float
