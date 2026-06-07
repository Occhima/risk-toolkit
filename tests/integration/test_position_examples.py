from __future__ import annotations

import polars as pl
import pytest
from schenberg.domain.schemas.position import (
    BookContract,
    InstrumentRisk,
    InstrumentValue,
    Position,
    ReportingFx,
)
from schenberg.position import book_value_rollup, position_risk, position_value


def test_position_value_book_rollup_and_delta_exposure() -> None:
    positions = Position.from_records(
        [
            {
                "position_id": "P1",
                "book": "BOOK-A",
                "instrument_type": "FORWARD",
                "instrument_id": "FWD-1",
                "quantity": 100.0,
                "side": 1.0,
                "unit_notional": 1.0,
            },
            {
                "position_id": "P2",
                "book": "BOOK-A",
                "instrument_type": "FORWARD",
                "instrument_id": "FWD-2",
                "quantity": 40.0,
                "side": -1.0,
                "unit_notional": 1.0,
            },
        ]
    )
    values = InstrumentValue.from_records(
        [
            {
                "instrument_type": "FORWARD",
                "instrument_id": "FWD-1",
                "value": 10.0,
                "currency": "USD",
            },
            {
                "instrument_type": "FORWARD",
                "instrument_id": "FWD-2",
                "value": 5.0,
                "currency": "USD",
            },
        ]
    )
    risks = InstrumentRisk.from_records(
        [
            {
                "instrument_type": "FORWARD",
                "instrument_id": "FWD-1",
                "currency": "USD",
                "delta": 0.95,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho": 0.0,
            },
            {
                "instrument_type": "FORWARD",
                "instrument_id": "FWD-2",
                "currency": "USD",
                "delta": 0.90,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho": 0.0,
            },
        ]
    )
    book = BookContract.from_records(
        [
            {
                "book": "BOOK-A",
                "desk": "Commodities",
                "legal_entity": "LE",
                "reporting_currency": "BRL",
            }
        ]
    )
    fx = ReportingFx.from_records(
        [{"currency": "USD", "reporting_currency": "BRL", "book_fx": 0.20}]
    )

    pos_values = position_value(positions, value=values, book=book, fx=fx).collect()
    by_id = {row["position_id"]: row for row in pos_values.to_dicts()}
    assert by_id["P1"]["mtm"] == pytest.approx(1000.0)
    assert by_id["P2"]["mtm"] == pytest.approx(-200.0)

    rollup = book_value_rollup.compute(pos_values.lazy()).collect()
    assert rollup.select("mtm").item() == pytest.approx(800.0)
    assert rollup.select("reported_mtm").item() == pytest.approx(4000.0)

    pos_risk = position_risk(positions, risk=risks).collect()
    risk_by_id = {row["position_id"]: row for row in pos_risk.to_dicts()}
    assert risk_by_id["P1"]["position_delta"] == pytest.approx(95.0)
    assert risk_by_id["P2"]["position_delta"] == pytest.approx(-36.0)

    assert "side" in Position.to_schema().columns
    assert "side" not in InstrumentValue.to_schema().columns
    assert isinstance(position_value(positions, value=values, book=book, fx=fx), pl.LazyFrame)
