from __future__ import annotations

from pathlib import Path
from typing import cast

import polars as pl
import pytest
from schenberg.domain.schemas.position import BookContract, InstrumentValue, Position, ReportingFx
from schenberg.position import position_value


def test_position_layer_owns_reporting_currency_conversion() -> None:
    instrument_value = 12.5
    values = InstrumentValue.from_records(
        [
            {
                "instrument_type": "FORWARD",
                "instrument_id": "FWD-1",
                "value": instrument_value,
                "currency": "USD",
            }
        ]
    )
    positions = Position.from_records(
        [
            {
                "position_id": "P1",
                "book": "B1",
                "instrument_type": "FORWARD",
                "instrument_id": "FWD-1",
                "quantity": 4.0,
                "side": -1.0,
                "unit_notional": None,
            }
        ]
    )
    book = BookContract.from_records(
        [{"book": "B1", "desk": "D", "legal_entity": "LE", "reporting_currency": "BRL"}]
    )
    fx = ReportingFx.from_records(
        [{"currency": "USD", "reporting_currency": "BRL", "book_fx": 0.25}]
    )

    out = cast(pl.DataFrame, position_value(positions, value=values, book=book, fx=fx).collect())
    assert out.select("mtm").item() == pytest.approx(-50.0)
    assert out.select("reported_mtm").item() == pytest.approx(-200.0)
    assert cast(pl.DataFrame, values.collect()).select("value").item() == instrument_value


def test_pricing_modules_do_not_read_reporting_currency() -> None:
    root = Path(__file__).parents[2]
    pricing_text = "\n".join(p.read_text() for p in (root / "schenberg" / "pricing").rglob("*.py"))
    assert "reporting_currency" not in pricing_text
