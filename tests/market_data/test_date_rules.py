from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl
import pytest
from schenberg.market_data.date_rules import (
    constant_month_of_tenor_year,
    copy_date,
    first_day_of_tenor_month,
    start_of_tenor_year,
    with_date_rule,
)
from schenberg.market_data.requirements import contract
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.market import FIXINGS


def test_start_of_tenor_year_maps_to_jan_first() -> None:
    lf = pl.DataFrame({"tenor_date": [date(2029, 5, 15), date(2029, 12, 31)]}).lazy()
    result = cast(pl.DataFrame, lf.with_columns(start_of_tenor_year()).collect())
    assert result["fixing_date"].to_list() == [date(2029, 1, 1), date(2029, 1, 1)]


def test_start_of_tenor_year_custom_columns() -> None:
    lf = pl.DataFrame({"tenor": [date(2031, 8, 20)]}).lazy()
    result = cast(
        pl.DataFrame,
        lf.with_columns(
            start_of_tenor_year(tenor_col="tenor", output_col="pca_fixing_date")
        ).collect(),
    )
    assert result["pca_fixing_date"].to_list() == [date(2031, 1, 1)]


def test_first_day_of_tenor_month() -> None:
    lf = pl.DataFrame({"tenor_date": [date(2028, 7, 22)]}).lazy()
    result = cast(pl.DataFrame, lf.with_columns(first_day_of_tenor_month()).collect())
    assert result["fixing_date"].to_list() == [date(2028, 7, 1)]


def test_constant_month_of_tenor_year_april() -> None:
    lf = pl.DataFrame({"tenor_date": [date(2028, 6, 1)]}).lazy()
    result = cast(
        pl.DataFrame,
        lf.with_columns(constant_month_of_tenor_year(month=4)).collect(),
    )
    assert result["fixing_date"].to_list() == [date(2028, 4, 1)]


def test_constant_month_of_tenor_year_rejects_invalid_month() -> None:
    with pytest.raises(ValueError, match="month must be between 1 and 12"):
        constant_month_of_tenor_year(month=0)
    with pytest.raises(ValueError, match="month must be between 1 and 12"):
        constant_month_of_tenor_year(month=13)


def test_copy_date() -> None:
    lf = pl.DataFrame({"payment_date": [date(2027, 3, 15)]}).lazy()
    result = cast(
        pl.DataFrame,
        lf.with_columns(copy_date(source_col="payment_date", output_col="discount_date")).collect(),
    )
    assert result["discount_date"].to_list() == [date(2027, 3, 15)]


def test_with_date_rule_is_a_thin_wrapper() -> None:
    lf = pl.DataFrame({"tenor_date": [date(2030, 9, 1)]}).lazy()
    result = cast(
        pl.DataFrame,
        with_date_rule(lf, start_of_tenor_year(output_col="pca_fixing_date")).collect(),
    )
    assert result["pca_fixing_date"].to_list() == [date(2030, 1, 1)]


def test_fixings_read_with_a_derived_date_key() -> None:
    req = FIXINGS.value().by(date=contract.pca_fixing_date).finalize("pca_factor")

    assert req.table == "fixings"
    assert req.left_keys == ("id_indexador", "pca_fixing_date")
    assert req.right_keys == ("id_indexador", "fixing_date")
    assert req.outputs == {"fixing_value": "pca_factor"}


def test_fixings_read_default_keys() -> None:
    req = FIXINGS.value().finalize("fixing_value")

    assert req.left_keys == ("id_indexador", "fixing_date")
    assert req.right_keys == ("id_indexador", "fixing_date")
    assert req.outputs == {"fixing_value": "fixing_value"}


def test_derive_fixing_date_attach_and_stay_lazy() -> None:
    """Derive pca_fixing_date via date rule, attach a fixing, assert correct value."""
    legs = pl.DataFrame(
        {
            "id_indexador": [20],
            "tenor_date": [date(2029, 5, 15)],
        }
    ).lazy()

    fixing_data = pl.DataFrame(
        {
            "id_indexador": [20],
            "fixing_date": [date(2029, 1, 1)],
            "fixing_value": [142.5],
        }
    ).lazy()

    snapshot = MarketSnapshot.from_sources(
        as_of=date(2026, 6, 3),
        sources=[MarketSource("fixings", fixing_data)],
    )

    prepared = legs.with_columns(
        start_of_tenor_year(tenor_col="tenor_date", output_col="pca_fixing_date")
    )

    req = FIXINGS.value().by(date=contract.pca_fixing_date).finalize("pca_factor")

    result_lf = snapshot.attach(prepared, req)

    assert isinstance(result_lf, pl.LazyFrame), "must remain lazy until collected"

    result = cast(pl.DataFrame, result_lf.collect())
    assert result["pca_fixing_date"].to_list() == [date(2029, 1, 1)]
    assert result["pca_factor"].to_list() == pytest.approx([142.5])
