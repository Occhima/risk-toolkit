from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl
import pytest
from schenberg.domain.schemas.market_data import FixingContract
from schenberg.market_data.date_rules import (
    constant_month_of_tenor_year,
    copy_date,
    energy_settlement_date,
    first_day_of_tenor_month,
    start_of_tenor_year,
    with_date_rule,
)
from schenberg.market_data.fixings import Fixings, FixingsSpec
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource


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


def test_energy_settlement_date_sixth_business_day_after_month_end() -> None:
    # 2029-06 ends Sat 2029-06-30; weekends only -> 6th business day is 2029-07-09.
    # 2026-07 ends Fri 2026-07-31 -> 2026-08-10.
    lf = pl.DataFrame({"delivery_period": ["2029-06", "2026-07"]}).lazy()
    result = cast(pl.DataFrame, lf.with_columns(energy_settlement_date()).collect())
    assert result["fixing_date"].to_list() == [date(2029, 7, 9), date(2026, 8, 10)]


def test_energy_settlement_date_honours_holidays() -> None:
    # Inserting an ANBIMA-style holiday inside the counting window pushes the
    # fixing one business day further out.
    lf = pl.DataFrame({"delivery_period": ["2026-07"]}).lazy()
    result = cast(
        pl.DataFrame,
        lf.with_columns(energy_settlement_date(holidays=[date(2026, 8, 5)])).collect(),
    )
    assert result["fixing_date"].to_list() == [date(2026, 8, 11)]


def test_energy_settlement_date_custom_columns_and_offset() -> None:
    lf = pl.DataFrame({"period": ["2026-07"]}).lazy()
    result = cast(
        pl.DataFrame,
        lf.with_columns(
            energy_settlement_date(
                period_col="period",
                output_col="settle_date",
                business_days_after_month_end=1,
            )
        ).collect(),
    )
    # 1st business day after Fri 2026-07-31 is Mon 2026-08-03.
    assert result["settle_date"].to_list() == [date(2026, 8, 3)]


def test_energy_settlement_date_rejects_non_positive_offset() -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        energy_settlement_date(business_days_after_month_end=0)


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


def test_fixings_spec_value_creates_correct_requirement() -> None:
    req = FixingsSpec().value(date_col="pca_fixing_date", output="pca_factor")

    assert req.table == "fixings"
    assert req.left_keys == ("id_indexador", "pca_fixing_date")
    assert req.right_keys == ("id_indexador", "fixing_date")
    assert req.outputs == {"fixing_value": "pca_factor"}


def test_fixings_spec_value_default_keys() -> None:
    req = FixingsSpec().value()

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

    req = FixingsSpec().value(
        indexer_col="id_indexador",
        date_col="pca_fixing_date",
        output="pca_factor",
    )

    result_lf = snapshot.attach(prepared, req)

    assert isinstance(result_lf, pl.LazyFrame), "must remain lazy until collected"

    result = cast(pl.DataFrame, result_lf.collect())
    assert result["pca_fixing_date"].to_list() == [date(2029, 1, 1)]
    assert result["pca_factor"].to_list() == pytest.approx([142.5])


def test_fixings_build_source_and_spec() -> None:
    data = FixingContract.from_records(
        [{"id_indexador": 1, "fixing_date": date(2026, 1, 1), "fixing_value": 100.0}]
    )
    fixings = Fixings.build(data=data, name="my_fixings")

    assert fixings.source().name == "my_fixings"
    assert fixings.spec().name == "my_fixings"
