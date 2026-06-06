"""A tiny stdlib of reusable, **typed** position measures.

These are the position-layer analogue of ``sum_`` / ``first_`` for a
:class:`~schenberg.core.fold.Fold`: small, declarative, composable building
blocks, not a framework. Every column is referenced by a *typed* schema column
(``cols(Position).side``, ``cols(InstrumentValue).value``) — never a bare string —
so a typo fails at construction, against the schema, not at ``collect``. Each
helper returns a :class:`~schenberg.position.view.Measure` that registers exactly
one term on a view as an :class:`~schenberg.core.expr.Expr`, resolving its inputs
through ``view.col(...)`` so it works regardless of how the sources were named.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import reduce

from schenberg.core.columns import ColumnLike, col_name, cols
from schenberg.core.expr import Expr, abs_
from schenberg.domain.schemas.position import (
    InstrumentValue,
    Position,
    PositionValue,
    ReportingFx,
)
from schenberg.position.view import Measure, PositionView

# Canonical typed column references for the common position vocabulary.
_POS = cols(Position)
_IV = cols(InstrumentValue)
_FX = cols(ReportingFx)
_PV = cols(PositionValue)


def exposure(
    *,
    side: ColumnLike = _POS.side,
    quantity: ColumnLike = _POS.quantity,
    name: ColumnLike = _PV.exposure,
) -> Measure:
    """Economic exposure: ``side * quantity``. Direction enters here."""

    def register(view: PositionView) -> Expr:
        return view.let(name, view.col(side) * view.col(quantity), symbol="E")

    return Measure(register)


def position_notional(
    *,
    quantity: ColumnLike = _POS.quantity,
    unit_notional: ColumnLike = _POS.unit_notional,
    name: ColumnLike = _PV.position_notional,
) -> Measure:
    """Gross notional held: ``|quantity| * unit_notional``."""

    def register(view: PositionView) -> Expr:
        return view.let(name, abs_(view.col(quantity)) * view.col(unit_notional), symbol="N")

    return Measure(register)


def scaled(
    column: ColumnLike, *, by: ColumnLike = _PV.exposure, name: ColumnLike | None = None
) -> Measure:
    """A position-scaled quantity: ``by * column`` (``exposure * column``).

    The one primitive behind every "lift a pure per-instrument quantity onto a
    position" measure — ``mtm`` is ``scaled(IV.value, name=PV.mtm)``, a position
    Greek is ``scaled(InstrumentRisk.delta)``. Defaults ``name`` to
    ``position_<column>``.
    """
    measure_name: ColumnLike = name if name is not None else f"position_{col_name(column)}"

    def register(view: PositionView) -> Expr:
        return view.let(measure_name, view.col(by) * view.col(column))

    return Measure(register)


def mtm(
    *,
    exposure: ColumnLike = _PV.exposure,
    value: ColumnLike = _IV.value,
    name: ColumnLike = _PV.mtm,
) -> Measure:
    """Mark-to-market of the position: ``exposure * instrument_value``."""

    def register(view: PositionView) -> Expr:
        return view.let(name, view.col(exposure) * view.col(value), symbol="MTM")

    return Measure(register)


def reported_mtm(
    *,
    mtm: ColumnLike = _PV.mtm,
    rate: ColumnLike = _FX.book_fx,
    name: ColumnLike = _PV.reported_mtm,
) -> Measure:
    """MTM converted into the book's reporting currency: ``mtm / book_fx``."""

    def register(view: PositionView) -> Expr:
        return view.let(name, view.col(mtm) / view.col(rate), symbol=r"\widehat{MTM}")

    return Measure(register)


def risk_factor(
    factor: ColumnLike, *, exposure: ColumnLike = _PV.exposure, prefix: str = "position_"
) -> Measure:
    """Lift one pure per-instrument risk factor onto a position:
    ``exposure * <factor>`` -> ``<prefix><factor>`` (e.g. ``position_delta``)."""
    return scaled(factor, by=exposure, name=f"{prefix}{col_name(factor)}")


def pnl_component(
    value: ColumnLike,
    *,
    name: ColumnLike,
    exposure: ColumnLike = _PV.exposure,
    rate: ColumnLike = _FX.book_fx,
) -> Measure:
    """One reported PnL-explain component, lifted from the instrument decomposition:
    ``exposure * <value> / book_fx`` -> ``<name>``."""

    def register(view: PositionView) -> Expr:
        return view.let(name, view.col(exposure) * view.col(value) / view.col(rate))

    return Measure(register)


def total(components: Iterable[ColumnLike], *, name: ColumnLike) -> Measure:
    """A total measure: the sum of the named component measures. Additive by
    construction, so ``total == Σ components``."""
    comps = list(components)

    def register(view: PositionView) -> Expr:
        terms = [view.col(component) for component in comps]
        summed = reduce(lambda a, b: a + b, terms)
        return view.let(name, summed, symbol="Σ")

    return Measure(register)
