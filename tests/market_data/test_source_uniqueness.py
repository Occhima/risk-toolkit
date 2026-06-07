from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from schenberg.market_data.objects.errors import DuplicateMarketKeyError
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource


def test_duplicate_curve_quote_keys_raise() -> None:
    source = MarketSource(
        "curves",
        pl.DataFrame(
            {"id_indexador": ["IDX", "IDX"], "tenor_days": [252, 252], "zero_rate": [0.1, 0.2]}
        ).lazy(),
        unique_by=("id_indexador", "tenor_days"),
    )
    with pytest.raises(DuplicateMarketKeyError) as err:
        MarketSnapshot.from_sources(as_of=date(2026, 6, 5), sources=[source])
    message = str(err.value)
    assert "curves" in message
    assert "id_indexador" in message
    assert "tenor_days" in message
    assert "duplicate" in message


def test_unique_source_passes() -> None:
    snap = MarketSnapshot.from_sources(
        as_of=date(2026, 6, 5),
        sources=[
            MarketSource(
                "curves",
                pl.DataFrame(
                    {"id_indexador": ["IDX"], "tenor_days": [252], "zero_rate": [0.1]}
                ).lazy(),
                unique_by=("id_indexador", "tenor_days"),
            )
        ],
    )
    assert snap.source("curves").name == "curves"


def test_source_without_unique_by_does_not_validate() -> None:
    snap = MarketSnapshot.from_sources(
        as_of=date(2026, 6, 5),
        sources=[
            MarketSource(
                "curves",
                pl.DataFrame({"id_indexador": ["IDX", "IDX"], "tenor_days": [252, 252]}).lazy(),
            )
        ],
    )
    assert snap.source("curves").unique_by == ()


def test_validation_happens_at_snapshot_construction_only(monkeypatch: pytest.MonkeyPatch) -> None:
    source = MarketSource(
        "curves",
        pl.DataFrame({"id_indexador": ["IDX"], "tenor_days": [252]}).lazy(),
        unique_by=("id_indexador", "tenor_days"),
    )
    calls = 0

    def validate() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(MarketSource, "validate_unique_keys", lambda self: validate())
    snap = MarketSnapshot.from_sources(as_of=date(2026, 6, 5), sources=[source])
    assert calls == 1
    snap.source("curves")
    snap.source("curves")
    assert calls == 1


def test_snapshot_build_default_skips_unique_key_validation() -> None:
    market = (
        MarketSnapshot.at(date(2026, 6, 6))
        .source(
            "curves",
            pl.DataFrame(
                {
                    "curve": ["BRL_DI", "BRL_DI"],
                    "tenor_days": [252, 252],
                    "zero_rate": [0.10, 0.11],
                }
            ),
            unique_by=("curve", "tenor_days"),
        )
        .build()
    )
    assert isinstance(market, MarketSnapshot)


def test_snapshot_build_validate_true_detects_duplicates() -> None:
    with pytest.raises(DuplicateMarketKeyError):
        (
            MarketSnapshot.at(date(2026, 6, 6))
            .source(
                "curves",
                pl.DataFrame(
                    {
                        "curve": ["BRL_DI", "BRL_DI"],
                        "tenor_days": [252, 252],
                        "zero_rate": [0.10, 0.11],
                    }
                ),
                unique_by=("curve", "tenor_days"),
            )
            .build(validate=True)
        )


def test_market_validate_detects_duplicates_after_build() -> None:
    market = (
        MarketSnapshot.at(date(2026, 6, 6))
        .source(
            "curves",
            pl.DataFrame(
                {
                    "curve": ["BRL_DI", "BRL_DI"],
                    "tenor_days": [252, 252],
                    "zero_rate": [0.10, 0.11],
                }
            ),
            unique_by=("curve", "tenor_days"),
        )
        .build()
    )
    with pytest.raises(DuplicateMarketKeyError):
        market.validate()
