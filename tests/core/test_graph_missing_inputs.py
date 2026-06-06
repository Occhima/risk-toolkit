from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl
import pytest
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.instruments.derivatives.forwards.formulas import forward_formula


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


def test_compute_raises_on_missing_contract_input() -> None:
    trades = _trades().drop("strike")
    with pytest.raises(ValueError) as err:
        forward_formula.compute(trades, market=_market(), view="output")
    message = str(err.value)
    assert "graph 'forward'" in message
    assert "view 'output'" in message
    assert "strike" in message
    assert "stage(..., allow_missing=True)" in message


def test_compute_raises_on_missing_market_join_key() -> None:
    trades = _trades().drop("payment_days")
    with pytest.raises(ValueError) as err:
        forward_formula.compute(trades, market=_market(), view="output")
    assert "graph 'forward'" in str(err.value)
    assert "payment_days" in str(err.value)


def test_stage_allow_missing_false_raises() -> None:
    with pytest.raises(ValueError, match="strike"):
        forward_formula._g.stage(_trades().drop("strike"), market=_market(), view="output")


def test_stage_allow_missing_true_adds_null_debug_columns() -> None:
    staged = forward_formula._g.stage(
        _trades().drop("strike"), market=_market(), view="output", allow_missing=True
    )
    out = cast(pl.DataFrame, staged.select("strike").collect())
    assert out["strike"].to_list() == [None]
