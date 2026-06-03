from __future__ import annotations

from typing import cast

import polars as pl
import pytest
from schenberg.pricing.api import price_swap


def test_price_swap_returns_one_row_per_swap_with_existing_sign_convention(
    swap_legs, swap_market
) -> None:
    result = cast(pl.DataFrame, price_swap(swap_legs, swap_market).collect())

    assert result.select("swap_id").item() == "SWP-1"
    assert result.height == 1
    assert result.select("ativo_pv").item() == pytest.approx(108_580.490164, rel=1e-6)
    assert result.select("passivo_pv").item() == pytest.approx(-77_239.829269, rel=1e-6)
    assert result.select("npv").item() == pytest.approx(31_340.660895, rel=1e-6)
