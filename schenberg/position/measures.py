"""A tiny stdlib of reusable position measures.

These are the position-layer analogue of ``sum_`` / ``first_`` for a
:class:`~schenberg.core.fold.Fold`: small, declarative, composable building
blocks, not a framework. Each returns a :class:`~schenberg.position.view.Measure`
that registers exactly one term on a view — so the common measures read as data
(``view.add(M.exposure(), M.mtm(), M.reported_mtm())``) while bespoke math still
drops straight to ``@view.measure`` and ``pl.Expr``. A measure resolves the
columns it needs through ``view.col(...)``, so it works regardless of how the
sources were named.
"""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from schenberg.core.graph import uses
from schenberg.position.view import Measure, PositionView


def exposure(*, side: str = "side", quantity: str = "quantity", name: str = "exposure") -> Measure:
    """Economic exposure: ``side * quantity``. Direction enters here."""

    def register(view: PositionView):
        @view.measure(name=name, symbol="E")
        def _(s=uses(view.col(side)), q=uses(view.col(quantity))) -> pl.Expr:
            return s * q

        return _

    return Measure(register)


def position_notional(
    *,
    quantity: str = "quantity",
    unit_notional: str = "unit_notional",
    name: str = "position_notional",
) -> Measure:
    """Gross notional held: ``|quantity| * unit_notional``."""

    def register(view: PositionView):
        @view.measure(name=name, symbol="N")
        def _(q=uses(view.col(quantity)), un=uses(view.col(unit_notional))) -> pl.Expr:
            return q.abs() * un

        return _

    return Measure(register)


def scaled(column: str, *, by: str = "exposure", name: str | None = None) -> Measure:
    """A position-scaled quantity: ``by * column`` (``exposure * column``).

    The one primitive behind every "lift a pure per-instrument quantity onto a
    position" measure — ``mtm`` is ``scaled("value", name="mtm")``, a position
    Greek is ``scaled("delta", name="position_delta")``. Defaults ``name`` to
    ``position_<column>``.
    """
    measure_name = name or f"position_{column}"

    def register(view: PositionView):
        @view.measure(name=measure_name)
        def _(factor=uses(view.col(by)), x=uses(view.col(column))) -> pl.Expr:
            return factor * x

        return _

    return Measure(register)


def mtm(*, exposure: str = "exposure", value: str = "value", name: str = "mtm") -> Measure:
    """Mark-to-market of the position: ``exposure * instrument_value``."""

    def register(view: PositionView):
        @view.measure(name=name, symbol="MTM")
        def _(e=uses(view.col(exposure)), v=uses(view.col(value))) -> pl.Expr:
            return e * v

        return _

    return Measure(register)


def risk_factor(factor: str, *, exposure: str = "exposure", prefix: str = "position_") -> Measure:
    """Lift one pure per-instrument risk factor onto a position:
    ``exposure * <factor>`` -> ``<prefix><factor>`` (e.g. ``position_delta``)."""
    return scaled(factor, by=exposure, name=f"{prefix}{factor}")


def reported_mtm(*, mtm: str = "mtm", rate: str = "book_fx", name: str = "reported_mtm") -> Measure:
    """MTM converted into the book's reporting currency: ``mtm / book_fx``."""

    def register(view: PositionView):
        @view.measure(name=name, symbol=r"\widehat{MTM}")
        def _(m=uses(view.col(mtm)), fx=uses(view.col(rate))) -> pl.Expr:
            return m / fx

        return _

    return Measure(register)


def pnl_component(component: str, *, exposure: str = "exposure", rate: str = "book_fx") -> Measure:
    """One reported PnL-explain component, lifted from the instrument decomposition:
    ``exposure * <component>_value_pnl / book_fx`` -> ``<component>_mtm_pnl``."""

    def register(view: PositionView):
        @view.measure(name=f"{component}_mtm_pnl")
        def _(
            e=uses(view.col(exposure)),
            c=uses(view.col(f"{component}_value_pnl")),
            fx=uses(view.col(rate)),
        ) -> pl.Expr:
            return e * c / fx

        return _

    return Measure(register)


def total(
    components: Iterable[str], *, suffix: str = "_mtm_pnl", name: str = "total_mtm_pnl"
) -> Measure:
    """A total measure: the sum of the named component measures. Additive by
    construction, so ``total_mtm_pnl == Σ <component>_mtm_pnl``."""

    comps = list(components)

    def register(view: PositionView):
        terms = [view.col(f"{c}{suffix}") for c in comps]

        def reduce_sum(*exprs: pl.Expr) -> pl.Expr:
            acc = exprs[0]
            for expr in exprs[1:]:
                acc = acc + expr
            return acc

        return view.derive(name, terms, reduce_sum, symbol="Σ")

    return Measure(register)
