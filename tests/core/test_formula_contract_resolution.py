"""Formula contract resolution: pa.check_types triggers rule injection."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

import pandera.polars as pa
import polars as pl
import pytest
from pandera.typing.polars import LazyFrame

from schenberg.core.graph import Formula, uses
from schenberg.domain.base import SchenbergDataFrameModel
from schenberg.domain.rules import rule_for
from schenberg.market_data import date_rules as dates


class IndexerEnum(StrEnum):
    CPI = "CPI"
    IPCA = "IPCA"


class ForwardContractPricing(SchenbergDataFrameModel):
    tenor: date
    indexer: IndexerEnum
    index_fixing_date: date

    @rule_for("index_fixing_date", selector="indexer", value=IndexerEnum.CPI)
    def _cpi(cls):  # noqa: N805
        return dates.add_days("tenor", 5)

    @rule_for("index_fixing_date", selector="indexer", default=True)
    def _default(cls):  # noqa: N805
        return dates.same_day("tenor")


class ForwardPricing(SchenbergDataFrameModel):
    value: float


class EnergyForwardPricing(ForwardContractPricing):
    submarket: str
    incentive: str
    strike: float


formula = Formula("dummy_forward", input=EnergyForwardPricing)
c = formula.input


@formula.formula()
def value(strike: pl.Expr = uses(c.strike)) -> pl.Expr:
    return strike


formula.returns("output", ForwardPricing, value=value)


@pa.check_types(lazy=True)
def price_dummy_forward(
    trades: LazyFrame[EnergyForwardPricing],
) -> LazyFrame[ForwardPricing]:
    return formula.compute(trades, view="output")


def test_formula_resolve_through_check_types() -> None:
    tenor = date(2026, 3, 10)
    lf = pl.DataFrame(
        {
            "tenor": [tenor],
            "indexer": [IndexerEnum.CPI],
            "submarket": ["SE"],
            "incentive": ["I1"],
            "strike": [42.0],
        }
    ).lazy()

    result = price_dummy_forward(lf)

    assert isinstance(result, pl.LazyFrame)
    df = result.collect()
    assert df["value"][0] == pytest.approx(42.0)


def test_output_is_lazy() -> None:
    tenor = date(2026, 3, 10)
    lf = pl.DataFrame(
        {
            "tenor": [tenor],
            "indexer": [IndexerEnum.IPCA],
            "submarket": ["S"],
            "incentive": ["I"],
            "strike": [10.0],
        }
    ).lazy()

    result = price_dummy_forward(lf)
    assert isinstance(result, pl.LazyFrame)
