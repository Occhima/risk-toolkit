from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import cast

import polars as pl
import pytest
from schenberg.domain.schemas.position import BookContract, Position, ReportingFx
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.position import position_value
from schenberg.pricing.api import forward_instrument_value, price_forward


def _market() -> MarketSnapshot:
    return MarketSnapshot.from_sources(
        as_of=date(2026, 6, 5),
        sources=[
            MarketSource(
                "curves",
                pl.DataFrame(
                    {
                        "id_indexador": ["IDX"],
                        "tenor_days": [252],
                        "forward_rate": [110.0],
                        "risk_free_rate": [0.10],
                    }
                ).lazy(),
                unique_by=("id_indexador", "tenor_days"),
            )
        ],
    )


def _trades() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "instrument_id": ["FWD-1"],
            "tenor": [date(2027, 6, 5)],
            "indexer": ["IDX"],
            "currency": ["USD"],
            "strike": [100.0],
            "payment_days": [252],
        }
    ).lazy()


def _pv() -> float:
    return (110.0 - 100.0) * math.exp(-0.10)


def test_price_forward_does_not_require_fx_rates_source() -> None:
    out = cast(pl.DataFrame, price_forward(_trades(), _market()).collect())
    assert out.select("present_value").item() == pytest.approx(_pv())
    assert out.select("value").item() == pytest.approx(_pv())


def test_forward_instrument_value_returns_own_currency() -> None:
    out = cast(pl.DataFrame, forward_instrument_value(_trades(), _market()).collect())
    assert out.select("currency").item() == "USD"
    assert out.select("value").item() == pytest.approx(_pv())


def test_position_value_applies_reporting_fx_once() -> None:
    values = forward_instrument_value(_trades(), _market())
    positions = Position.from_records(
        [
            {
                "position_id": "P1",
                "book": "B1",
                "instrument_type": "FORWARD",
                "instrument_id": "FWD-1",
                "quantity": 10.0,
                "side": 1.0,
                "unit_notional": None,
            }
        ]
    )
    book = BookContract.from_records(
        [{"book": "B1", "desk": "D", "legal_entity": "LE", "reporting_currency": "BRL"}]
    )
    fx1 = ReportingFx.from_records(
        [{"currency": "USD", "reporting_currency": "BRL", "book_fx": 0.2}]
    )
    fx2 = ReportingFx.from_records(
        [{"currency": "USD", "reporting_currency": "BRL", "book_fx": 0.4}]
    )
    out1 = cast(pl.DataFrame, position_value(positions, value=values, book=book, fx=fx1).collect())
    out2 = cast(pl.DataFrame, position_value(positions, value=values, book=book, fx=fx2).collect())
    assert out1.select("mtm").item() == pytest.approx(10.0 * _pv())
    assert out2.select("mtm").item() == pytest.approx(10.0 * _pv())
    assert out1.select("reported_mtm").item() == pytest.approx(10.0 * _pv() / 0.2)
    assert out2.select("reported_mtm").item() == pytest.approx(10.0 * _pv() / 0.4)


def test_no_own_currency_hack_in_runtime_code() -> None:
    root = Path(__file__).parents[5]
    runtime = "\n".join(p.read_text() for p in (root / "schenberg").rglob("*.py"))
    assert "_OWN" + "_CURRENCY" not in runtime
