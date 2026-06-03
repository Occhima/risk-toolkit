from __future__ import annotations

from datetime import date, timedelta

import pytest
from schenberg.market_data.calendar import ANBIMA, ANBIMA_HOLIDAYS, anbima_calendar
from schenberg.market_data.calendar.anbima import (
    CAL_PATH,
    generate_holidays,
    holidays_for_year,
    load_holidays,
    write_cal_file,
)

_BUSINESS_DAYS_PER_YEAR = 252

# Known 2026 ANBIMA holidays: fixed national dates plus the Easter-derived
# moveable feasts (Easter 2026 = Apr 5) and Consciência Negra (Nov 20, since 2024).
EXPECTED_2026 = {
    date(2026, 1, 1),  # Confraternização Universal
    date(2026, 2, 16),  # Carnival Monday
    date(2026, 2, 17),  # Carnival Tuesday
    date(2026, 4, 3),  # Good Friday
    date(2026, 4, 21),  # Tiradentes
    date(2026, 5, 1),  # Dia do Trabalho
    date(2026, 6, 4),  # Corpus Christi
    date(2026, 9, 7),  # Independência
    date(2026, 10, 12),  # N. Sra Aparecida
    date(2026, 11, 2),  # Finados
    date(2026, 11, 15),  # Proclamação da República
    date(2026, 11, 20),  # Consciência Negra
    date(2026, 12, 25),  # Natal
}


def test_holidays_for_year_matches_known_2026() -> None:
    assert holidays_for_year(2026) == EXPECTED_2026


def test_consciencia_negra_only_from_2024() -> None:
    assert date(2023, 11, 20) not in holidays_for_year(2023)
    assert date(2024, 11, 20) in holidays_for_year(2024)


def test_bundled_calendar_file_is_loaded() -> None:
    assert CAL_PATH.exists()
    assert load_holidays() == ANBIMA_HOLIDAYS
    assert EXPECTED_2026.issubset(ANBIMA_HOLIDAYS)
    assert ANBIMA.holidays == ANBIMA_HOLIDAYS
    assert ANBIMA.business_days_per_year == _BUSINESS_DAYS_PER_YEAR


def test_anbima_calendar_reads_from_file() -> None:
    assert anbima_calendar().holidays == ANBIMA_HOLIDAYS


def test_write_and_load_round_trip(tmp_path) -> None:
    path = tmp_path / "ANBIMA.cal"
    write_cal_file(path, start_year=2025, end_year=2027)
    loaded = load_holidays(path)
    assert loaded == generate_holidays(2025, 2027)
    assert EXPECTED_2026.issubset(loaded)


def test_cal_file_format_is_comments_then_iso_dates() -> None:
    lines = CAL_PATH.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("#")
    data_lines = [ln for ln in lines if ln and not ln.startswith("#")]
    # Every non-comment line parses as an ISO date and the file is sorted.
    parsed = [date.fromisoformat(ln) for ln in data_lines]
    assert parsed == sorted(parsed)


def test_generate_holidays_covers_requested_range() -> None:
    start, end = 2030, 2031
    holidays = generate_holidays(start, end)
    assert all(start <= d.year <= end for d in holidays)
    assert date(start, 1, 1) in holidays


@pytest.mark.parametrize(
    ("year", "easter"),
    [(2024, date(2024, 3, 31)), (2026, date(2026, 4, 5)), (2027, date(2027, 3, 28))],
)
def test_good_friday_two_days_before_easter(year: int, easter: date) -> None:
    assert (easter - timedelta(days=2)) in holidays_for_year(year)
