from __future__ import annotations

from datetime import date
from typing import cast

import pandas as pd
import polars as pl
import pytest
from pandera.typing.polars import LazyFrame
from schenberg.core.columns import ColumnRef, cols
from schenberg.core.graph import ExprGraph
from schenberg.core.router import Router
from schenberg.domain.enums import InstrumentType
from schenberg.domain.schemas.forward import EnergyForwardLeg, ForwardTrade
from schenberg.domain.schemas.market_data import (
    DiCurveContract,
    EnergyForwardCurveContract,
    FxRatesContract,
)
from schenberg.domain.schemas.position import Position, PricedPosition
from schenberg.market_data.calendar.conventions import Calendar
from schenberg.market_data.curves.di import DiCurve, DiCurveSpec
from schenberg.market_data.forwards import EnergyForwardCurveSpec
from schenberg.market_data.fx import FxRates
from schenberg.market_data.shocks import ParallelZeroRateShock
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.position.functions import (
    pnl_from_priced_positions,
    price_forward_instruments,
    with_prices,
)
from schenberg.position.pipelines import valuation_pipe


def test_dataframe_model_ergonomic_constructors() -> None:
    records = [{"currency": "BRL", "fx_rate": 1.0}]

    assert (
        cast(pl.DataFrame, FxRatesContract.from_records(records).collect()).select("fx_rate").item()
        == 1.0
    )
    assert (
        cast(
            pl.DataFrame, FxRatesContract.from_vectors(currency=["BRL"], fx_rate=[1.0]).collect()
        ).height
        == 1
    )
    assert (
        cast(pl.DataFrame, FxRatesContract.from_pandas(pd.DataFrame(records)).collect()).height == 1
    )


def test_column_ref_and_schema_cols() -> None:
    F = cols(ForwardTrade)

    assert isinstance(F.instrument_id, ColumnRef)
    assert F.instrument_id.name == "instrument_id"
    assert pl.DataFrame({"instrument_id": ["A"]}).select(F.instrument_id.expr()).item() == "A"
    assert (F.forward_family == "ENERGY").expr().meta.root_names() == ["forward_family"]
    with pytest.raises(AttributeError):
        _ = F.quantity


def test_di_curve_build_source_and_spec() -> None:
    calendar = Calendar.business_252(set())
    data = DiCurveContract.from_records(
        [{"curve_name": "DI", "id_indexador": 1, "tenor_days": 21, "zero_rate": 0.1}]
    )

    curve = DiCurve.build(data=data, calendar=calendar)
    source = curve.source()
    req = curve.spec().zero_rate()

    assert source.name == "di_curve"
    assert source.schema is DiCurveContract
    assert req.table == "di_curve"
    assert req.left_keys == ("id_indexador", "payment_days")
    assert req.right_keys == ("id_indexador", "tenor_days")
    assert req.outputs == {"zero_rate": "zero_rate"}


def test_energy_forward_curve_spec_requirement() -> None:
    req = EnergyForwardCurveSpec().forward_price()

    assert req.table == "energy_forward_curve"
    assert req.left_keys == ("submarket", "delivery_period")
    assert req.right_keys == ("submarket", "delivery_period")
    assert req.outputs == {"forward_price": "forward_price", "settle_days": "payment_days"}


def test_market_snapshot_from_sources_attach_and_shock() -> None:
    source = MarketSource(
        "di_curve",
        pl.DataFrame(
            {"curve_name": ["DI"], "id_indexador": [1], "tenor_days": [21], "zero_rate": [0.1]}
        ).lazy(),
        DiCurveContract,
    )
    snapshot = MarketSnapshot.from_sources(as_of=date(2026, 6, 3), sources=[source])
    trades = pl.DataFrame({"id_indexador": [1], "payment_days": [21]}).lazy()

    attached = snapshot.attach(trades, DiCurveSpec().zero_rate(output="rate"))
    bumped = ParallelZeroRateShock(shift=0.01).apply(snapshot)

    assert snapshot.source("di_curve") == source
    expected_rate = 0.1
    assert cast(pl.DataFrame, attached.collect()).select("rate").item() == expected_rate
    assert bumped.source("di_curve").schema is DiCurveContract
    assert cast(pl.DataFrame, bumped.source("di_curve").data.collect()).select(
        "zero_rate"
    ).item() == pytest.approx(0.11)


def test_router_fallback_receives_all_rows_with_no_cases() -> None:
    graph = ExprGraph("fallback")

    @graph.node()
    def priced_marker(payment_days: pl.Expr) -> pl.Expr:
        return payment_days * 0 + 1

    graph.with_outputs("pricing", priced_marker="priced_marker")
    router = Router.by(cols(ForwardTrade).forward_family).default(graph)

    out = router.compute_for(pl.DataFrame({"forward_family": ["A"], "payment_days": [2]}).lazy())

    assert cast(pl.DataFrame, out.collect()).select("priced_marker").item() == 1


def test_energy_forward_graph_and_position_layer(energy_inputs, energy_market) -> None:
    priced = price_forward_instruments(energy_inputs, energy_market)
    positions = Position.from_records(
        [
            {
                "position_id": "POS-1",
                "book": "Energy Desk",
                "instrument_type": InstrumentType.FORWARD.value,
                "instrument_id": "ENG-1",
                "quantity": 100.0,
            }
        ]
    )
    valued = positions.pipe(with_prices, priced)

    priced_df = cast(pl.DataFrame, priced.collect())
    valued_df = cast(pl.DataFrame, valued.collect())

    assert priced_df.height == 1
    assert priced_df.select("instrument_id").item() == "ENG-1"
    assert priced_df.select("price").item() == pytest.approx(49.057467, rel=1e-6)
    assert valued_df.select("mtm").item() == pytest.approx(4905.7467, rel=1e-6)


def test_pnl_from_priced_positions_can_be_called_independently() -> None:
    today = PricedPosition.from_records(
        [
            {
                "position_id": "P",
                "book": "B",
                "instrument_type": "FORWARD",
                "instrument_id": "I",
                "quantity": 2.0,
                "price": 6.0,
                "mtm": 12.0,
            }
        ]
    )
    previous = PricedPosition.from_records(
        [
            {
                "position_id": "P",
                "book": "B",
                "instrument_type": "FORWARD",
                "instrument_id": "I",
                "quantity": 2.0,
                "price": 5.0,
                "mtm": 10.0,
            }
        ]
    )

    out = pnl_from_priced_positions(today, previous).collect()

    expected_pnl = 2.0
    assert cast(pl.DataFrame, out).select("mtm_pnl").item() == expected_pnl


def test_valuation_pipe_exposes_intermediate_outputs(energy_inputs, energy_market) -> None:
    positions = Position.from_records(
        [
            {
                "position_id": "POS-1",
                "book": "Energy Desk",
                "instrument_type": "FORWARD",
                "instrument_id": "ENG-1",
                "quantity": 100.0,
            }
        ]
    )

    env = valuation_pipe.run(forwards=energy_inputs, positions=positions, market=energy_market)

    assert {"forward_prices", "prices", "priced_positions", "book_mtm"}.issubset(env)
    assert env["book_mtm"].collect().select("mtm").item() == pytest.approx(4905.7467, rel=1e-6)


def test_expected_user_workflow() -> None:
    calendar = Calendar.business_252(holidays=set())
    di_curve_data = DiCurveContract.from_records(
        [{"curve_name": "DI", "id_indexador": 1, "tenor_days": 21, "zero_rate": 0.1}]
    )
    energy_curve_data = EnergyForwardCurveContract.from_records(
        [
            {
                "submarket": "SE",
                "delivery_period": "2026-07",
                "forward_price": 260.0,
                "settle_days": 21,
            }
        ]
    )
    fx_data = FxRatesContract.from_records([{"currency": "BRL", "fx_rate": 1.0}])

    di_curve = DiCurve.build(data=di_curve_data, calendar=calendar, name="di_curve")
    energy_curve_source = MarketSource(
        "energy_forward_curve", energy_curve_data, EnergyForwardCurveContract
    )
    fx_rates = FxRates.build(data=fx_data, name="fx_rates")
    market = MarketSnapshot.from_sources(
        as_of=date(2026, 6, 3),
        sources=[di_curve.source(), energy_curve_source, fx_rates.source()],
    )
    forwards = EnergyForwardLeg.from_records(
        [
            {
                "instrument_id": "ENE-001",
                "instrument_type": "FORWARD",
                "forward_family": "ENERGY",
                "settlement_type": "PHYSICAL",
                "currency": "BRL",
                "id_indexador": 1,
                "payment_days": 21,
                "submarket": "SE",
                "delivery_period": "2026-07",
                "buy_sell": "BUY",
                "strike": 250.0,
            }
        ]
    )
    positions = Position.from_records(
        [
            {
                "position_id": "POS-001",
                "book": "Energy Desk",
                "instrument_type": "FORWARD",
                "instrument_id": "ENE-001",
                "quantity": 100.0,
            }
        ]
    )

    prices = price_forward_instruments(cast(LazyFrame[ForwardTrade], forwards), market)
    priced_positions = positions.pipe(with_prices, prices)

    assert cast(pl.DataFrame, priced_positions.collect()).select("mtm").item() == pytest.approx(
        cast(pl.DataFrame, prices.collect()).select("price").item() * 100.0
    )
