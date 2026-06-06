"""Contract-rule injection through SchenbergDataFrameModel.validate()."""

from __future__ import annotations

import ast
from datetime import date
from enum import StrEnum
from pathlib import Path

import pandera.polars as pa
import polars as pl
import pytest
from pandera.typing.polars import LazyFrame
from schenberg.domain.base import SchenbergDataFrameModel
from schenberg.domain.rules import rule_for
from schenberg.market_data import date_rules as dates


class IndexerEnum(StrEnum):
    CPI = "CPI"
    IPCA = "IPCA"
    PLD = "PLD"


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


@pa.check_types(lazy=True)
def identity_contract(
    df: LazyFrame[ForwardContractPricing],
) -> LazyFrame[ForwardContractPricing]:
    return df


# ---- Test 1: missing column is created ------------------------------------


def test_missing_column_filled_by_rule() -> None:
    tenor_cpi = date(2026, 3, 10)
    tenor_ipca = date(2026, 4, 15)

    lf = pl.DataFrame(
        {
            "tenor": [tenor_cpi, tenor_ipca],
            "indexer": [IndexerEnum.CPI, IndexerEnum.IPCA],
        }
    ).lazy()

    result = identity_contract(lf).collect()

    assert result["index_fixing_date"][0] == date(2026, 3, 15)  # +5
    assert result["index_fixing_date"][1] == tenor_ipca  # same day


# ---- Test 2: non-null explicit values preserved ---------------------------


def test_explicit_non_null_value_preserved() -> None:
    explicit = date(2026, 1, 1)
    tenor = date(2026, 3, 10)

    lf = pl.DataFrame(
        {
            "tenor": [tenor],
            "indexer": [IndexerEnum.CPI],
            "index_fixing_date": [explicit],
        }
    ).lazy()

    result = identity_contract(lf).collect()
    assert result["index_fixing_date"][0] == explicit


# ---- Test 3: null rows filled, non-null rows preserved --------------------


def test_null_filled_non_null_preserved() -> None:
    tenor = date(2026, 3, 10)
    explicit = date(2026, 1, 1)

    lf = pl.DataFrame(
        {
            "tenor": [tenor, tenor],
            "indexer": [IndexerEnum.CPI, IndexerEnum.CPI],
            "index_fixing_date": [None, explicit],
        },
        schema={
            "tenor": pl.Date,
            "indexer": pl.String,
            "index_fixing_date": pl.Date,
        },
    ).lazy()

    result = identity_contract(lf).collect()
    assert result["index_fixing_date"][0] == date(2026, 3, 15)  # +5 (was null)
    assert result["index_fixing_date"][1] == explicit  # preserved


# ---- Test 4: child class overrides parent rule ----------------------------


class ChildSchemaPlus10(ForwardContractPricing):
    @rule_for("index_fixing_date", selector="indexer", value=IndexerEnum.CPI)
    def _cpi(cls):  # noqa: N805
        return dates.add_days("tenor", 10)


@pa.check_types(lazy=True)
def identity_child(
    df: LazyFrame[ChildSchemaPlus10],
) -> LazyFrame[ChildSchemaPlus10]:
    return df


def test_child_overrides_parent_cpi_rule() -> None:
    tenor = date(2026, 3, 10)
    lf = pl.DataFrame({"tenor": [tenor], "indexer": [IndexerEnum.CPI]}).lazy()

    result = identity_child(lf).collect()
    assert result["index_fixing_date"][0] == date(2026, 3, 20)  # +10


# ---- Test 5: multiple mixins compose without conflict ---------------------


class CurrencyEnum(StrEnum):
    BRL = "BRL"
    EUR = "EUR"


class CurrencyFixingMixin(SchenbergDataFrameModel):
    currency: CurrencyEnum
    currency_fixing_date: date

    @rule_for("currency_fixing_date", selector="currency", value=CurrencyEnum.EUR)
    def _eur(cls):  # noqa: N805
        return dates.previous_day("tenor")

    @rule_for("currency_fixing_date", selector="currency", default=True)
    def _default(cls):  # noqa: N805
        return dates.same_day("tenor")


class FullForwardContract(
    ForwardContractPricing,
    CurrencyFixingMixin,
    SchenbergDataFrameModel,
):
    currency: CurrencyEnum
    strike: float


@pa.check_types(lazy=True)
def identity_full(
    df: LazyFrame[FullForwardContract],
) -> LazyFrame[FullForwardContract]:
    return df


def test_multiple_mixins_compose() -> None:
    tenor = date(2026, 3, 10)
    lf = pl.DataFrame(
        {
            "tenor": [tenor, tenor],
            "indexer": [IndexerEnum.CPI, IndexerEnum.IPCA],
            "currency": [CurrencyEnum.EUR, CurrencyEnum.BRL],
            "strike": [100.0, 200.0],
        }
    ).lazy()

    result = identity_full(lf).collect()

    # index_fixing_date
    assert result["index_fixing_date"][0] == date(2026, 3, 15)  # CPI +5
    assert result["index_fixing_date"][1] == tenor  # IPCA same day

    # currency_fixing_date
    assert result["currency_fixing_date"][0] == date(2026, 3, 9)  # EUR -1
    assert result["currency_fixing_date"][1] == tenor  # BRL same day


# ---- Test 6: no .collect() in runtime files -------------------------------


def test_no_collect_in_runtime_files() -> None:
    root = Path(__file__).parents[2] / "schenberg" / "domain"
    for fname in ("base.py", "rules.py"):
        src = (root / fname).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "collect":
                # allow collect_schema
                if isinstance(node.value, ast.Attribute) and node.value.attr == "collect_schema":
                    continue
                # allow node names like collect_schema
                pytest.fail(f"{fname}: .collect() call found at line {node.lineno}")
