"""The built-in forward pricers emit pure InstrumentValue, ready for position_value."""

from __future__ import annotations

import math
from datetime import date
from typing import cast

import polars as pl
import pytest
from schenberg.domain.schemas.position import (
    BookContract,
    Position,
    ReportingFx,
)
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.position import position_value
from schenberg.pricing.api import forward_instrument_value


def _market(fx_rate: float) -> MarketSnapshot:
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
            ),
            MarketSource(
                "fx_rates",
                pl.DataFrame({"currency": ["USD"], "fx_rate": [fx_rate]}).lazy(),
            ),
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


def _expected_present_value() -> float:
    future_value = 110.0 - 100.0
    return future_value * math.exp(-0.10 * 1.0)


def test_forward_instrument_value_is_lazy() -> None:
    out = forward_instrument_value(_trades(), _market(5.0))
    assert isinstance(out, pl.LazyFrame)


def test_forward_instrument_value_is_pure_own_currency() -> None:
    # value is the pre-FX present value; the contract currency string survives
    # (it is NOT clobbered by the forward formula's fx_rate market read).
    out = cast(pl.DataFrame, forward_instrument_value(_trades(), _market(5.0)).collect())
    assert out.columns == ["instrument_type", "instrument_id", "value", "currency"]
    assert out.select("currency").item() == "USD"
    assert out.select("value").item() == pytest.approx(_expected_present_value())
    assert out.select("instrument_type").item() == "FORWARD"


def test_instrument_type_is_configurable() -> None:
    out = cast(
        pl.DataFrame,
        forward_instrument_value(_trades(), _market(5.0), instrument_type="FX_FORWARD").collect(),
    )
    assert out.select("instrument_type").item() == "FX_FORWARD"


def test_forward_value_feeds_position_value_without_a_manual_rename() -> None:
    values = forward_instrument_value(_trades(), _market(5.0))
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
    # USD -> BRL conversion lives in the position layer, applied exactly once.
    fx = ReportingFx.from_records(
        [{"currency": "USD", "reporting_currency": "BRL", "book_fx": 0.2}]
    )

    out = cast(pl.DataFrame, position_value(positions, value=values, book=book, fx=fx).collect())
    pv = _expected_present_value()
    assert out.select("mtm").item() == pytest.approx(10.0 * pv)  # own currency (USD)
    assert out.select("reported_mtm").item() == pytest.approx(10.0 * pv / 0.2)  # BRL
