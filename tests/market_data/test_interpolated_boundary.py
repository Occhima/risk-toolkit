from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl
import pytest
from schenberg.market_data.interpolated import InterpolatedBook, InterpolatedSpec
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.market import VOL


def test_interpolated_attach_returns_lazy_frame() -> None:
    snapshot = MarketSnapshot.from_sources(
        as_of=date(2026, 6, 3),
        sources=[
            MarketSource(
                "rates_curve",
                pl.DataFrame(
                    {
                        "id_indexador": [1, 1],
                        "tenor_days": [126, 252],
                        "zero_rate": [0.10, 0.12],
                    }
                ).lazy(),
                unique_by=("id_indexador", "tenor_days"),
            )
        ],
    )
    req = InterpolatedSpec("rates_curve", axes=("tenor_days",)).value(
        "zero_rate", output="rate", on=("payment_days",)
    )
    attached = snapshot.attach(
        pl.DataFrame({"id_indexador": [1], "payment_days": [189]}).lazy(), req
    )
    assert isinstance(attached, pl.LazyFrame)
    assert cast(pl.DataFrame, attached.collect())["rate"].to_list() == pytest.approx([0.11])


def test_quote_collection_isolated_to_interpolation_book(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    original = InterpolatedBook.from_quotes

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(InterpolatedBook, "from_quotes", counted)
    snapshot = MarketSnapshot.from_sources(
        as_of=date(2026, 6, 3),
        sources=[
            MarketSource(
                "rates_curve",
                pl.DataFrame(
                    {"id_indexador": [1, 1], "tenor_days": [126, 252], "zero_rate": [0.1, 0.2]}
                ).lazy(),
            )
        ],
    )
    req = InterpolatedSpec("rates_curve", axes=("tenor_days",)).value(
        "zero_rate", output="rate", on=("payment_days",)
    )
    attached = snapshot.attach(
        pl.DataFrame({"id_indexador": [1], "payment_days": [189]}).lazy(), req
    )
    assert isinstance(attached, pl.LazyFrame)
    assert calls == 1


def test_missing_interpolation_group_raises_when_collected() -> None:
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
    attached = snapshot.attach(trades, VOL.implied_vol().finalize("vol"))

    with pytest.raises(ValueError, match="unknown group"):
        attached.collect()
