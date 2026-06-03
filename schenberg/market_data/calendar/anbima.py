"""The ANBIMA holiday calendar — Brazil's financial-market business-day set.

The BUS/252 day count that drives Brazilian fixed-income and energy discounting
skips weekends plus the national holidays *and* the Easter-derived moveable
feasts (Carnival Monday & Tuesday, Good Friday, Corpus Christi). Consciência
Negra (Nov 20) joins as a national holiday from 2024 (Lei 14.759/2023).

The holiday dates live in a sibling ``ANBIMA.cal`` data file (one ISO date per
line); :func:`load_holidays` reads it and :data:`ANBIMA_HOLIDAYS` /
:data:`ANBIMA` expose it as a ``frozenset`` and a :class:`Calendar`. The file is
regenerated from :func:`generate_holidays` via :func:`write_cal_file`.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from schenberg.market_data.calendar.conventions import Calendar

__all__ = [
    "ANBIMA",
    "ANBIMA_HOLIDAYS",
    "CAL_PATH",
    "anbima_calendar",
    "generate_holidays",
    "load_holidays",
    "write_cal_file",
]

CAL_PATH = Path(__file__).with_name("ANBIMA.cal")

# Fixed-date national holidays (month, day).
_FIXED_HOLIDAYS = (
    (1, 1),  # Confraternização Universal
    (4, 21),  # Tiradentes
    (5, 1),  # Dia do Trabalho
    (9, 7),  # Independência
    (10, 12),  # Nossa Senhora Aparecida
    (11, 2),  # Finados
    (11, 15),  # Proclamação da República
    (12, 25),  # Natal
)
_CONSCIENCIA_NEGRA_FROM = 2024  # Nov 20 became a national holiday in 2024.

_DEFAULT_START_YEAR = 2000
_DEFAULT_END_YEAR = 2100


def _easter_sunday(year: int) -> date:
    """Easter Sunday via the anonymous Gregorian algorithm (Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = (h + ell - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def holidays_for_year(year: int) -> set[date]:
    """The ANBIMA holiday set for a single year (fixed dates + moveable feasts)."""
    holidays = {date(year, month, day) for month, day in _FIXED_HOLIDAYS}
    if year >= _CONSCIENCIA_NEGRA_FROM:
        holidays.add(date(year, 11, 20))  # Consciência Negra
    easter = _easter_sunday(year)
    holidays.add(easter - timedelta(days=48))  # Carnival Monday
    holidays.add(easter - timedelta(days=47))  # Carnival Tuesday
    holidays.add(easter - timedelta(days=2))  # Good Friday (Sexta-feira Santa)
    holidays.add(easter + timedelta(days=60))  # Corpus Christi
    return holidays


def generate_holidays(
    start_year: int = _DEFAULT_START_YEAR,
    end_year: int = _DEFAULT_END_YEAR,
) -> frozenset[date]:
    """All ANBIMA holidays across ``[start_year, end_year]`` (inclusive)."""
    holidays: set[date] = set()
    for year in range(start_year, end_year + 1):
        holidays |= holidays_for_year(year)
    return frozenset(holidays)


_CAL_HEADER = (
    "# ANBIMA — Brazilian financial-market holiday calendar (BUS/252 basis).",
    "# National holidays plus Easter-derived moveable feasts (Carnival Monday &",
    "# Tuesday, Good Friday, Corpus Christi); Consciência Negra (Nov 20) from 2024.",
    "# Weekends (Sat/Sun) are non-business days. Regenerate with",
    "# schenberg.market_data.calendar.anbima.write_cal_file.",
    "# Format: '#' comments or one ISO-8601 date (YYYY-MM-DD) per line.",
)


def write_cal_file(
    path: Path = CAL_PATH,
    *,
    start_year: int = _DEFAULT_START_YEAR,
    end_year: int = _DEFAULT_END_YEAR,
) -> None:
    """Materialize the calendar to ``path`` as ``ANBIMA.cal``."""
    dates = sorted(generate_holidays(start_year, end_year))
    lines = [*_CAL_HEADER, *(d.isoformat() for d in dates)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_holidays(path: Path = CAL_PATH) -> frozenset[date]:
    """Read the holiday dates from a ``.cal`` file (comments and blanks ignored)."""
    holidays: set[date] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        holidays.add(date.fromisoformat(line))
    return frozenset(holidays)


def anbima_calendar(path: Path = CAL_PATH) -> Calendar:
    """A BUS/252 :class:`Calendar` backed by the ANBIMA holidays in ``path``."""
    return Calendar.business_252(load_holidays(path))


# Loaded once from the bundled data file — the project-wide ANBIMA calendar.
# The file is generated and committed; regenerate it on the fly if it is absent
# (e.g. a fresh checkout extending the year range) so import never hard-fails.
if not CAL_PATH.exists():
    write_cal_file()
ANBIMA_HOLIDAYS: frozenset[date] = load_holidays()
ANBIMA: Calendar = Calendar.business_252(ANBIMA_HOLIDAYS)
