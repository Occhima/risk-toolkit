from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl
from schenberg.core.columns import ColumnSet
from schenberg.core.market import curve, energy_forward, fixing, fx
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource


def test_column_set_exposes_left_and_right_keys() -> None:
    keys = ColumnSet.from_pairs(("left_a", "right_a"), ("left_b", "right_b"))

    assert keys.left_keys == ("left_a", "left_b")
    assert keys.right_keys == ("right_a", "right_b")


def test_market_requirement_constructors_use_column_sets() -> None:
    assert curve("zero_rate").left_keys == ("id_indexador", "payment_days")
    assert curve("zero_rate").right_keys == ("id_indexador", "tenor_days")
    assert fixing().outputs == {"fixing_value": "base_index"}
    assert energy_forward().outputs == {
        "forward_price": "forward_price",
        "settle_days": "payment_days",
    }
    assert fx().left_keys == ("currency",)


def test_market_snapshot_attach_lazily_joins_and_renames_outputs() -> None:
    snapshot = MarketSnapshot.from_sources(
        as_of=date(2026, 6, 3),
        sources=[
            MarketSource(
                "curves",
                pl.DataFrame({"id_indexador": [1], "tenor_days": [252], "zero_rate": [0.1]}).lazy(),
            )
        ],
    )
    trades = pl.DataFrame({"id_indexador": [1], "payment_days": [252]}).lazy()

    attached = snapshot.attach(trades, curve(outputs={"zero_rate": "rate"}))

    assert isinstance(attached, pl.LazyFrame)
    expected_rate = 0.1
    out = cast(pl.DataFrame, attached.collect())
    assert out.select("rate").item() == expected_rate


def test_market_snapshot_from_sources_indexes_sources() -> None:
    source = MarketSource("curves", pl.DataFrame({"x": [1]}).lazy())
    snapshot = MarketSnapshot.from_sources(as_of=date(2026, 6, 3), sources=[source])

    assert snapshot.source("curves") == source
