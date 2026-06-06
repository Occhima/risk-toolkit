"""The position-computation layer.

Pricing answers *what is one unit worth?* (a pure ``InstrumentValue``); the
position layer answers *how much do I hold, and what is that worth in my book's
terms?*. A :class:`~schenberg.position.view.PositionView` lifts a pure instrument
value onto a position and a book/reporting context, exposing the measures
(exposure, notional, mtm, reported mtm; PnL-explain components and total) by name.
Book/portfolio roll-up is a *later* layer — a :class:`~schenberg.core.fold.Fold`,
not part of the view.
"""

from __future__ import annotations

from schenberg.position import measures
from schenberg.position.view import Measure, PositionView
from schenberg.position.views import (
    book_value_rollup,
    position_pnl_explain,
    position_value,
)

__all__ = [
    "Measure",
    "PositionView",
    "book_value_rollup",
    "measures",
    "position_pnl_explain",
    "position_value",
]
