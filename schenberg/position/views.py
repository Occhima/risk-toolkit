"""The built-in position views: ``position_value``, ``position_pnl_explain`` and
``position_risk``.

These are concrete :class:`~schenberg.position.view.PositionView` declarations —
callable (``position_value(positions, value=..., book=..., fx=...)``) and
inspectable (``position_value.explain()``). They are the position-layer analogue
of the built-in pricers, and the *same* declaration regardless of what pure
per-instrument quantity is being lifted: a single ``value`` (``mtm``), a PnL
decomposition (``*_mtm_pnl``), or a vector of risk factors (``position_<greek>``).
"""

from __future__ import annotations

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

# ---- position value: exposure, notional, mtm, reported mtm -------------------

position_value = (
    PositionView("position_value", output=PositionValue)
    .spine(Position)
    .source("value", InstrumentValue, on=("instrument_type", "instrument_id"))
    .source("book", BookContract, on="book")
    .source("fx", ReportingFx, on=("currency", "reporting_currency"))
    .add(
        M.exposure(),  # side * quantity
        M.position_notional(),  # |quantity| * unit_notional
        M.mtm(),  # exposure * value
        M.reported_mtm(),  # mtm / book_fx
    )
    .returns()
)


# ---- position PnL explain: per-component reported PnL, additive --------------

PNL_COMPONENTS = ("roll", "curve", "fx", "fixing", "residual")

position_pnl_explain = (
    PositionView("position_pnl_explain", output=PositionPnlExplain)
    .spine(Position)
    .source("pnl", InstrumentPnlExplain, on=("instrument_type", "instrument_id"))
    .source("book", BookContract, on="book")
    .source("fx", ReportingFx, on=("currency", "reporting_currency"))
    .add(
        M.exposure(),
        *[M.pnl_component(component) for component in PNL_COMPONENTS],
        M.total(PNL_COMPONENTS),  # total_mtm_pnl == Σ <component>_mtm_pnl
    )
    .returns()
)


# ---- position risk: each pure Greek lifted onto the position -----------------

RISK_FACTORS = ("delta", "gamma", "vega", "theta", "rho")

position_risk = (
    PositionView("position_risk", output=PositionRisk)
    .spine(Position)
    .source("risk", InstrumentRisk, on=("instrument_type", "instrument_id"))
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
    .by("book")
    .returns(
        None,
        exposure=sum_("exposure"),
        mtm=sum_("mtm"),
        reported_mtm=sum_("reported_mtm"),
    )
)
"""Group one-row-per-position measures into one-row-per-book totals. Aggregation
is a *later* layer than the position view and reuses the monoidal
:class:`~schenberg.core.fold.Fold`; a position is never an aggregate."""
