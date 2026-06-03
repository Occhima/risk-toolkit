from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl
import pytest
from schenberg.core.columns import ColumnSet
from schenberg.core.market import curve, fixing
from schenberg.domain.schemas.market_data import VolSurfaceContract
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.market_data.volatility import VolSurfaces, VolSurfaceSpec


def test_column_set_exposes_left_and_right_keys() -> None:
    keys = ColumnSet.from_pairs(("left_a", "right_a"), ("left_b", "right_b"))

    assert keys.left_keys == ("left_a", "left_b")
    assert keys.right_keys == ("right_a", "right_b")


def test_market_requirement_constructors_use_column_sets() -> None:
    assert curve("zero_rate").left_keys == ("id_indexador", "payment_days")
    assert curve("zero_rate").right_keys == ("id_indexador", "tenor_days")
    assert fixing().outputs == {"fixing_value": "base_index"}


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


def test_vol_surface_requirement_attaches_by_indexer_tenor_and_strike() -> None:
    quotes = VolSurfaceContract.from_polars(
        pl.DataFrame(
            {
                "id_indexador": [1, 1, 1, 1, 2, 2, 2, 2],
                "tenor_days": [252, 252, 504, 504, 252, 252, 504, 504],
                "strike": [100.0, 110.0, 100.0, 110.0, 100.0, 110.0, 100.0, 110.0],
                "implied_vol": [0.20, 0.22, 0.21, 0.23, 0.30, 0.32, 0.31, 0.33],
            }
        )
    )
    surfaces = VolSurfaces.build(quotes)
    source = surfaces.source()
    assert source.schema is VolSurfaceContract
    req = VolSurfaceSpec().implied_vol(output="vol")
    assert hasattr(req, "attach")
    snapshot = MarketSnapshot.from_sources(as_of=date(2026, 6, 3), sources=[source])
    trades = pl.DataFrame(
        {
            "id_indexador": [1, 2],
            "payment_days": [252, 252],
            "strike": [100.0, 100.0],
        }
    ).lazy()

    out = cast(pl.DataFrame, snapshot.attach(trades, req).collect())

    assert out["vol"].to_list() == [0.20, 0.30]


def test_vol_surface_requirement_unknown_indexer_raises() -> None:
    quotes = pl.DataFrame(
        {
            "id_indexador": [1, 1, 1, 1],
            "tenor_days": [252, 252, 504, 504],
            "strike": [100.0, 110.0, 100.0, 110.0],
            "implied_vol": [0.20, 0.22, 0.21, 0.23],
        }
    ).lazy()
    snapshot = MarketSnapshot.from_sources(
        as_of=date(2026, 6, 3), sources=[MarketSource("vol_surface", quotes)]
    )
    trades = pl.DataFrame({"id_indexador": [9], "payment_days": [252], "strike": [100.0]}).lazy()

    attached = snapshot.attach(trades, VolSurfaceSpec().implied_vol(output="vol"))

    with pytest.raises(ValueError, match="unknown id_indexador"):
        attached.collect()
