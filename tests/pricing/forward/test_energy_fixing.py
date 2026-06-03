from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl
from schenberg.core.columns import cols
from schenberg.domain.schemas.forward import EnergyForwardLeg
from schenberg.pricing.instruments.forward.energy import with_fixing_date

E = cols(EnergyForwardLeg)


def _legs(delivery_periods: list[str]) -> pl.LazyFrame:
    n = len(delivery_periods)
    return pl.DataFrame(
        {
            "instrument_id": [f"ENG-{i}" for i in range(n)],
            "instrument_type": ["FORWARD"] * n,
            "forward_family": ["ENERGY"] * n,
            "settlement_type": ["PHYSICAL"] * n,
            "currency": ["BRL"] * n,
            "id_indexador": [1] * n,
            "payment_days": [30] * n,
            "submarket": ["SE"] * n,
            "delivery_period": delivery_periods,
            "strike": [100.0] * n,
        }
    ).lazy()


def test_fixing_is_sixth_business_day_of_following_month() -> None:
    out = cast(
        pl.DataFrame,
        with_fixing_date(_legs(["2026-07", "2026-08"]))
        .select(E.delivery_period.name, E.fixing_date.name)
        .collect(),
    )

    fixings = dict(zip(out[E.delivery_period.name], out[E.fixing_date.name], strict=True))
    # July delivery settles on the 6th business day of August 2026 (Aug 1 is a Sat,
    # so business days run 3,4,5,6,7,10 -> the 6th is Aug 10).
    assert fixings["2026-07"] == date(2026, 8, 10)
    assert fixings["2026-08"] == date(2026, 9, 9)


def test_fixing_skips_anbima_holidays() -> None:
    # December delivery -> January, where Jan 1 (Confraternização) is an ANBIMA
    # holiday, so the 6th business day lands on Jan 11 rather than Jan 8.
    out = cast(pl.DataFrame, with_fixing_date(_legs(["2026-12"])).collect())

    assert out[E.fixing_date.name].item() == date(2027, 1, 11)
