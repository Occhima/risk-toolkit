"""Business-day counting — the same calculation two ways.

* :func:`business_day_count_expr` — a Polars expression over date columns,
  backed by :func:`polars.business_day_count`; use it inside ``with_columns``.
* :func:`business_day_count` — the eager NumPy equivalent
  (:func:`numpy.busday_count`) for plain dates / arrays.

Both count business days from ``start`` (inclusive) to ``end`` (exclusive),
skipping weekends and the ``holidays`` you pass (e.g. ``ANBIMA_HOLIDAYS``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date

import numpy as np
import polars as pl
from numpy.typing import NDArray

__all__ = ["business_day_count", "business_day_count_expr"]

# A single date or a 1-D batch of them, in any form NumPy can read as days.
Dates = date | str | Sequence[date | str] | NDArray[np.datetime64]


def business_day_count_expr(
    start: pl.Expr | str,
    end: pl.Expr | str,
    *,
    holidays: Iterable[date] = (),
) -> pl.Expr:
    """Business days in ``[start, end)`` as a Polars expression."""
    return pl.business_day_count(start, end, holidays=list(holidays))


def _as_days(values: Dates) -> NDArray[np.datetime64]:
    return np.asarray(values, dtype="datetime64[D]")


def business_day_count(
    start: Dates,
    end: Dates,
    *,
    holidays: Iterable[date] = (),
) -> NDArray[np.int64]:
    """Business days in ``[start, end)`` via NumPy — scalars or arrays of dates."""
    return np.busday_count(_as_days(start), _as_days(end), holidays=_as_days(sorted(holidays)))
