from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from schenberg.domain.base import SchenbergDataFrameModel
from schenberg.market_data.date_rules import add_days, same_day
from schenberg.market_data.roles import (
    Fixing,
    With,
    bind,
    market_role,
    roles_of,
)
from schenberg.market_data.snapshot import MarketSnapshot

# ---- role declarations -------------------------------------------------------

ForwardPrice = (
    market_role("forward_price")
    .read("curves", "forward_rate")
    .by(indexer="id_indexador", payment_days="tenor_days")
)

RiskFree = (
    market_role("risk_free")
    .read("curves", "risk_free_rate")
    .by(indexer="id_indexador", payment_days="tenor_days")
)

IndexerValue = (
    market_role("indexer_value")
    .read("fixings", "fixing_value")
    .by(indexer="id_indexador")
    .fixing(
        "fixing_date",
        Fixing.on("indexer").when("CPI", add_days("tenor", 5)).otherwise(same_day("tenor")),
    )
)


class ForwardPricingInput(
    With[ForwardPrice],
    With[RiskFree],
    With[IndexerValue],
    SchenbergDataFrameModel,
):
    instrument_id: str
    strike: float
    indexer: str
    payment_days: int
    tenor: date


# ---- fixtures ----------------------------------------------------------------


def _raw() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "instrument_id": ["A", "B"],
            "strike": [10.0, 20.0],
            "indexer": ["DI", "CPI"],
            "payment_days": [252, 504],
            "tenor": [date(2027, 1, 1), date(2028, 1, 1)],
        }
    ).lazy()


def _snapshot() -> MarketSnapshot:
    curves = pl.DataFrame(
        {
            "id_indexador": ["DI", "CPI"],
            "tenor_days": [252, 504],
            "forward_rate": [5.0, 6.0],
            "risk_free_rate": [0.10, 0.12],
        }
    )
    fixings = pl.DataFrame(
        {
            "id_indexador": ["DI", "CPI"],
            # DI: same_day(tenor); CPI: tenor + 5 days
            "fixing_date": [date(2027, 1, 1), date(2028, 1, 6)],
            "fixing_value": [100.0, 200.0],
        }
    )
    return (
        MarketSnapshot.at(date(2026, 6, 6))
        .source("curves", curves, unique_by=("id_indexador", "tenor_days"))
        .source("fixings", fixings, unique_by=("id_indexador", "fixing_date"))
        .build()
    )


# ---- Fixing ------------------------------------------------------------------


def test_fixing_direct_is_the_rule() -> None:
    out = (
        _raw()
        .with_columns(Fixing.rule(same_day("tenor")).expr().alias("fix"))
        .select("fix")
        .collect()
    )
    assert out["fix"].to_list() == [date(2027, 1, 1), date(2028, 1, 1)]


def test_fixing_conditional_branches_on_selector() -> None:
    fixing = Fixing.on("indexer").when("CPI", add_days("tenor", 5)).otherwise(same_day("tenor"))
    out = _raw().with_columns(fixing.expr().alias("fix")).select("fix").collect()
    # DI -> same day; CPI -> +5 days
    assert out["fix"].to_list() == [date(2027, 1, 1), date(2028, 1, 6)]


def test_fixing_conditional_without_otherwise_raises() -> None:
    with pytest.raises(ValueError, match="otherwise"):
        Fixing.on("indexer").when("CPI", same_day("tenor")).expr()


# ---- roles_of (anti-drift) ---------------------------------------------------


def test_roles_of_discovers_all_declared_roles() -> None:
    names = {role.name for role in roles_of(ForwardPricingInput)}
    assert names == {"forward_price", "risk_free", "indexer_value"}


def test_published_columns_subset_of_input_schema() -> None:
    published = {role.name for role in roles_of(ForwardPricingInput)}
    columns = set(ForwardPricingInput.to_schema().columns.keys())
    assert published <= columns


# ---- bind --------------------------------------------------------------------


def test_bind_joins_curves_and_fixing_and_validates() -> None:
    enriched = bind(_raw(), _snapshot(), ForwardPricingInput).collect()

    assert set(enriched.columns) == set(ForwardPricingInput.to_schema().columns.keys())
    # transient fixing key must not survive
    assert not any(c.startswith("__fix_") for c in enriched.columns)

    by_id = {row["instrument_id"]: row for row in enriched.to_dicts()}
    assert by_id["A"]["forward_price"] == 5.0
    assert by_id["A"]["risk_free"] == pytest.approx(0.10)
    assert by_id["A"]["indexer_value"] == 100.0  # DI same-day fixing matched
    assert by_id["B"]["indexer_value"] == 200.0  # CPI +5d fixing matched


def test_bind_stays_lazy() -> None:
    assert isinstance(bind(_raw(), _snapshot(), ForwardPricingInput), pl.LazyFrame)


# ---- snapshot builder --------------------------------------------------------


def test_builder_matches_from_sources() -> None:
    built = _snapshot()
    assert built.as_of == date(2026, 6, 6)
    assert set(built.sources) == {"curves", "fixings"}


def test_builder_validates_unique_keys() -> None:
    dupes = pl.DataFrame(
        {
            "id_indexador": ["DI", "DI"],
            "tenor_days": [252, 252],
            "forward_rate": [5.0, 5.0],
            "risk_free_rate": [0.1, 0.1],
        }
    )
    with pytest.raises(Exception):  # noqa: B017,PT011 — DuplicateMarketKeyError
        MarketSnapshot.at(date(2026, 6, 6)).source(
            "curves", dupes, unique_by=("id_indexador", "tenor_days")
        ).build()
