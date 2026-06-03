from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl
from schenberg.market_data.calendar import (
    ANBIMA_HOLIDAYS,
    business_day_count,
    business_day_count_expr,
)

WEEK = (date(2026, 6, 1), date(2026, 6, 8))  # Mon -> next Mon: 5 weekdays


def test_expr_counts_weekdays_inclusive_exclusive() -> None:
    # Mon 2026-06-01 .. Mon 2026-06-08 exclusive, weekends only -> 5 business days.
    lf = pl.DataFrame({"s": [date(2026, 6, 1)], "e": [date(2026, 6, 8)]}).lazy()
    result = cast(
        pl.DataFrame,
        lf.with_columns(business_day_count_expr("s", "e").alias("n")).collect(),
    )
    assert result["n"].to_list() == [5]


def test_expr_skips_holidays() -> None:
    lf = pl.DataFrame({"s": [date(2026, 6, 1)], "e": [date(2026, 6, 8)]}).lazy()
    result = cast(
        pl.DataFrame,
        lf.with_columns(
            business_day_count_expr("s", "e", holidays=[date(2026, 6, 3)]).alias("n")
        ).collect(),
    )
    assert result["n"].to_list() == [4]


def test_numpy_scalar_matches_polars() -> None:
    start, end = WEEK
    frame = pl.DataFrame({"s": [start], "e": [end]})
    assert (
        int(business_day_count(start, end))
        == frame.select(business_day_count_expr("s", "e")).item()
    )
    holidays = [date(2026, 6, 3)]
    assert (
        int(business_day_count(start, end, holidays=holidays))
        == frame.select(business_day_count_expr("s", "e", holidays=holidays)).item()
    )


def test_numpy_array_form() -> None:
    out = business_day_count(
        [date(2026, 6, 1), date(2026, 7, 1)],
        [date(2026, 6, 8), date(2026, 7, 8)],
    )
    assert out.tolist() == [5, 5]


def test_expr_and_numpy_agree_on_anbima_calendar() -> None:
    start, end = date(2026, 1, 2), date(2026, 12, 31)
    lf = pl.DataFrame({"s": [start], "e": [end]})
    expr_n = lf.select(business_day_count_expr("s", "e", holidays=ANBIMA_HOLIDAYS)).item()
    numpy_n = int(business_day_count(start, end, holidays=ANBIMA_HOLIDAYS))
    assert expr_n == numpy_n
    # 2026 has 13 ANBIMA holidays; a full year is well under the raw weekday count.
    raw_weekdays = int(business_day_count(start, end))
    assert numpy_n < raw_weekdays
