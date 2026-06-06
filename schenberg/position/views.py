"""The built-in position views: ``position_value``, ``position_pnl_explain`` and
``position_risk``.

These are concrete :class:`~schenberg.position.view.PositionView` declarations —
callable (``position_value(positions, value=..., book=..., fx=...)``) and
inspectable (``position_value.explain()``). They are the position-layer analogue
of the built-in pricers, and the *same* declaration regardless of what pure
per-instrument quantity is being lifted: a single ``value`` (``mtm``), a PnL
decomposition (``*_mtm_pnl``), or a vector of risk factors (``position_<greek>``).

Every column is referenced by a *typed* schema column — ``cols(InstrumentValue)``,
``cols(PositionPnlExplain)`` — so the declarations are checked against the schemas
at import time, not at ``collect()``.
"""

from __future__ import annotations

from schenberg.core.columns import cols
from schenberg.core.fold import Fold, sum_
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
from schenberg.position import measures as M
from schenberg.position.view import PositionView

# Typed column references — the position-layer vocabulary.
IV = cols(InstrumentValue)
IPE = cols(InstrumentPnlExplain)
IR = cols(InstrumentRisk)
PV = cols(PositionValue)
PPE = cols(PositionPnlExplain)
PR = cols(PositionRisk)

# ---- position value: exposure, notional, mtm, reported mtm -------------------

position_value = (
    PositionView("position_value", output=PositionValue)
    .spine(Position)
    .source("value", InstrumentValue, on=(IV.instrument_type, IV.instrument_id))
    .source("book", BookContract, on=cols(BookContract).book)
    .source("fx", ReportingFx, on=(IV.currency, cols(ReportingFx).reporting_currency))
    .add(
        M.exposure(),  # side * quantity
        M.position_notional(),  # |quantity| * unit_notional
        M.mtm(),  # exposure * value
        M.reported_mtm(),  # mtm / book_fx
    )
    .returns()
)


# ---- position PnL explain: per-component reported PnL, additive --------------

# Each pure instrument component (left) maps to its position measure (right).
PNL_COMPONENTS = (
    (IPE.roll_value_pnl, PPE.roll_mtm_pnl),
    (IPE.curve_value_pnl, PPE.curve_mtm_pnl),
    (IPE.fx_value_pnl, PPE.fx_mtm_pnl),
    (IPE.fixing_value_pnl, PPE.fixing_mtm_pnl),
    (IPE.residual_value_pnl, PPE.residual_mtm_pnl),
)

position_pnl_explain = (
    PositionView("position_pnl_explain", output=PositionPnlExplain)
    .spine(Position)
    .source("pnl", InstrumentPnlExplain, on=(IPE.instrument_type, IPE.instrument_id))
    .source("book", BookContract, on=cols(BookContract).book)
    .source("fx", ReportingFx, on=(IPE.currency, cols(ReportingFx).reporting_currency))
    .add(
        M.exposure(),
        *[M.pnl_component(value=src, name=out) for src, out in PNL_COMPONENTS],
        # total_mtm_pnl == Σ <component>_mtm_pnl
        M.total([out for _, out in PNL_COMPONENTS], name=PPE.total_mtm_pnl),
    )
    .returns()
)


# ---- position risk: each pure Greek lifted onto the position -----------------

RISK_FACTORS = (IR.delta, IR.gamma, IR.vega, IR.theta, IR.rho)

position_risk = (
    PositionView("position_risk", output=PositionRisk)
    .spine(Position)
    .source("risk", InstrumentRisk, on=(IR.instrument_type, IR.instrument_id))
    .add(
        M.exposure(),
        # each position_<greek> = exposure * <greek>
        *[M.risk_factor(factor) for factor in RISK_FACTORS],
    )
    .returns()
)
"""Lift the pure per-instrument Greeks onto positions. The view is identical in
shape to ``position_value`` — only the joined quantity differs. Reporting-currency
risk (for the currency-valued Greeks) is the same ``/ book_fx`` pattern as
``reported_mtm``: add ``book``/``fx`` sources and a ``reported_*`` measure."""


# ---- book roll-up: a Fold, the layer *after* the position view ---------------

book_value_rollup = (
    Fold("book_value_rollup", input_schema=PositionValue)
    .by(PV.book)
    .returns(
        None,
        exposure=sum_(PV.exposure),
        mtm=sum_(PV.mtm),
        reported_mtm=sum_(PV.reported_mtm),
    )
)
"""Group one-row-per-position measures into one-row-per-book totals. Aggregation
is a *later* layer than the position view and reuses the monoidal
:class:`~schenberg.core.fold.Fold`; a position is never an aggregate."""
