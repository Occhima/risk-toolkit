"""DV01: reprice under a +1bp rate bump and difference, reusing the forward pricer."""

from __future__ import annotations

import math
from datetime import date
from typing import cast

import polars as pl
import pytest
from schenberg.core.columns import cols
from schenberg.domain.schemas.position import InstrumentDv01, Position
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.position import measures as M
from schenberg.position.view import PositionView
from schenberg.pricing.api import forward_instrument_value
from schenberg.risk import Dv01Calculator


def _market(rate: float = 0.10) -> MarketSnapshot:
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
                        "risk_free_rate": [rate],
                    }
                ).lazy(),
            ),
            MarketSource(
                "fx_rates",
                pl.DataFrame({"currency": ["USD"], "fx_rate": [5.0]}).lazy(),
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


def test_dv01_matches_central_repricing() -> None:
    calc = Dv01Calculator.parallel(forward_instrument_value, column="risk_free_rate", bump=1e-4)
    out = cast(pl.DataFrame, calc.compute(_trades(), _market()).collect())

    assert out.columns == list(InstrumentDv01.to_schema().columns.keys())
    # present_value = (110 - 100) * exp(-r*T); dv01 = pv(r+1bp) - pv(r)
    pv = lambda r: (110.0 - 100.0) * math.exp(-r * 1.0)  # noqa: E731
    assert out.select("dv01").item() == pytest.approx(pv(0.10 + 1e-4) - pv(0.10), rel=1e-9)
    assert out.select("dv01").item() < 0.0  # rates up -> a positive forward value falls
    assert out.select("currency").item() == "USD"


def test_dv01_is_lazy() -> None:
    calc = Dv01Calculator.parallel(forward_instrument_value)
    assert isinstance(calc.compute(_trades(), _market()), pl.LazyFrame)


class PositionDv01(Position):
    position_dv01: float


def test_dv01_lifts_onto_a_position_like_any_risk_factor() -> None:
    # DV01 is a pure per-instrument quantity, so the position layer scales it by
    # exposure with the same M.scaled primitive used for mtm and the Greeks.
    dv01 = Dv01Calculator.parallel(forward_instrument_value).compute(_trades(), _market())

    view = (
        PositionView("position_dv01", output=PositionDv01)
        .spine(Position)
        .source("risk", InstrumentDv01, on=("instrument_type", "instrument_id"))
        .add(
            M.exposure(),
            M.scaled(cols(InstrumentDv01).dv01, name=cols(PositionDv01).position_dv01),
        )
        .returns()
    )
    positions = Position.from_records(
        [
            {
                "position_id": "P1",
                "book": "B",
                "instrument_type": "FORWARD",
                "instrument_id": "FWD-1",
                "quantity": 1_000_000.0,
                "side": -1.0,  # short: flips the sign of the position DV01
                "unit_notional": None,
            }
        ]
    )
    out = cast(pl.DataFrame, view(positions, risk=dv01).collect())
    instrument_dv01 = cast(pl.DataFrame, dv01.collect()).select("dv01").item()
    assert out.select("position_dv01").item() == pytest.approx(-1_000_000.0 * instrument_dv01)
