from __future__ import annotations

from datetime import date

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import LazyFrame
from schenberg.pricing.instruments.derivatives.forwards.contracts import (
    ForwardContractPricing,
)


@pa.check_types(lazy=True)
def identity_forward(
    trades: LazyFrame[ForwardContractPricing],
) -> LazyFrame[ForwardContractPricing]:
    return trades


def test_forward_contract_mixins_fill_missing_dates() -> None:
    trades = pl.DataFrame(
        {
            "instrument_id": ["FWD-1", "FWD-2"],
            "tenor": [date(2027, 1, 10), date(2027, 1, 10)],
            "indexer": ["CPI", "OTHER"],
            "currency": ["EUR", "BRL"],
            "strike": [100.0, 100.0],
            "payment_days": [252, 252],
        }
    ).lazy()

    result = identity_forward(trades)
    assert isinstance(result, pl.LazyFrame)

    got = result.collect()

    # CPI: tenor + 5 days
    assert got["index_fixing_date"][0] == date(2027, 1, 15)
    # OTHER: default → same day
    assert got["index_fixing_date"][1] == date(2027, 1, 10)

    # BRL: default → same day
    assert got["currency_fixing_date"][1] == date(2027, 1, 10)


def test_forward_contract_mixins_preserve_user_dates() -> None:
    trades = pl.DataFrame(
        {
            "instrument_id": ["FWD-1"],
            "tenor": [date(2027, 1, 10)],
            "indexer": ["CPI"],
            "currency": ["EUR"],
            "strike": [100.0],
            "payment_days": [252],
            "index_fixing_date": [date(2099, 1, 1)],
            "currency_fixing_date": [date(2099, 1, 2)],
        }
    ).lazy()

    got = identity_forward(trades).collect()

    assert got["index_fixing_date"][0] == date(2099, 1, 1)
    assert got["currency_fixing_date"][0] == date(2099, 1, 2)
