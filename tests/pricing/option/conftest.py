from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource

TENORS = [126, 252, 504]
STRIKES = [90.0, 100.0, 110.0]
# implied vol smile/term grid: vols[tenor][strike]
VOL_GRID = {
    (126, 90.0): 0.24, (126, 100.0): 0.21, (126, 110.0): 0.25,
    (252, 90.0): 0.22, (252, 100.0): 0.20, (252, 110.0): 0.23,
    (504, 90.0): 0.21, (504, 100.0): 0.19, (504, 110.0): 0.21,
}  # fmt: skip


def _flat_curve(col: str, value: float) -> pl.LazyFrame:
    return pl.DataFrame({"id_indexador": [1] * 3, "tenor_days": TENORS, col: [value] * 3}).lazy()


@pytest.fixture
def option_market() -> MarketSnapshot:
    vol_rows = {
        "id_indexador": [1] * len(VOL_GRID),
        "tenor_days": [t for t, _ in VOL_GRID],
        "strike": [k for _, k in VOL_GRID],
        "implied_vol": list(VOL_GRID.values()),
    }
    return MarketSnapshot.from_sources(
        as_of=date(2026, 6, 3),
        sources=[
            MarketSource("vol_surface", pl.DataFrame(vol_rows).lazy()),
            MarketSource("curves", _flat_curve("zero_rate", 0.10)),
            MarketSource("carry_curve", _flat_curve("cost_of_carry", 0.10)),  # b = r
            MarketSource("dividend_curve", _flat_curve("div_yield", 0.03)),
        ],
    )


def _option_rows() -> list[dict]:
    rows = []
    for model in ("GENERALIZED", "MERTON"):
        for kind in ("CALL", "PUT"):
            rows.append(
                {
                    "option_id": f"{model[0]}-{kind[0]}",
                    "instrument_type": "OPTION",
                    "option_model": model,
                    "option_kind": kind,
                    "id_indexador": 1,
                    "spot": 100.0,
                    "strike": 100.0,
                    "payment_days": 252,
                }
            )
    return rows


@pytest.fixture
def option_inputs() -> pl.LazyFrame:
    return pl.DataFrame(_option_rows()).lazy()
