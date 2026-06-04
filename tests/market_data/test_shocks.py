from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl
import pytest
from schenberg.market_data.path import MarketPath
from schenberg.market_data.shocks import Shock, curve_parallel_shift
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource


def _market() -> MarketSnapshot:
    return MarketSnapshot.from_sources(
        as_of=date(2026, 6, 3),
        sources=[
            MarketSource(
                "curves",
                pl.DataFrame(
                    {"id_indexador": [1], "tenor_days": [252], "zero_rate": [0.10]}
                ).lazy(),
                schema=object,
            )
        ],
    )


def _zero_rate(market: MarketSnapshot) -> float:
    return cast(pl.DataFrame, market.source("curves").data.collect()).select("zero_rate").item()


def test_market_path_get_focuses_a_source_column() -> None:
    market = _market()
    whole = MarketPath("curves").get(market)
    one = MarketPath("curves").column("zero_rate").get(market)

    assert "tenor_days" in whole.collect_schema().names()
    assert one.collect_schema().names() == ["zero_rate"]


def test_market_path_modify_builds_a_shock_that_returns_new_snapshot() -> None:
    market = _market()
    shock = MarketPath("curves").column("zero_rate").modify(lambda r: r + 1e-4)

    assert isinstance(shock, Shock)
    stressed = market.apply(shock)

    assert _zero_rate(stressed) == pytest.approx(0.1001)
    # original untouched
    assert _zero_rate(market) == pytest.approx(0.10)


def test_shock_preserves_source_schema() -> None:
    market = _market()
    stressed = market.apply(curve_parallel_shift(shift=0.01))

    assert stressed.source("curves").schema is market.source("curves").schema


def test_shocks_compose_in_order() -> None:
    market = _market()
    one = curve_parallel_shift(shift=0.01)
    two = curve_parallel_shift(shift=0.02)

    composed = Shock.compose(one, two)
    assert _zero_rate(market.apply(composed)) == pytest.approx(0.13)

    chained = one.then(two)
    assert _zero_rate(market.apply(chained)) == pytest.approx(0.13)


def test_identity_shock_is_a_noop() -> None:
    market = _market()
    assert _zero_rate(market.apply(Shock.identity())) == pytest.approx(0.10)


def test_shock_explains_itself() -> None:
    shock = curve_parallel_shift(shift=0.01)
    assert "Shock" in shock.explain()
    assert "endomorphism" in shock.explain()
    assert shock.info()["name"]


def test_market_path_set_eagerly_modifies() -> None:
    market = _market()
    stressed = MarketPath("curves").column("zero_rate").set_(market, lambda r: r * 2)

    assert _zero_rate(stressed) == pytest.approx(0.20)
    assert _zero_rate(market) == pytest.approx(0.10)
