from __future__ import annotations

import math
from datetime import date
from typing import cast

import polars as pl
import pytest
from pandera.typing.polars import LazyFrame
from schenberg.domain.schemas import SwapInput
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.api import price_swap


def test_swap_leg_in_foreign_currency_converts_by_fx_rate(swap_market) -> None:
    """A leg priced in a foreign currency is just FX market data: declare the
    rate and the leg's PV is translated into the reporting currency. A leg whose
    currency has no quote (or none at all) stays local at rate 1.0."""
    market = swap_market.with_source(
        MarketSource("fx_rates", pl.DataFrame({"currency": ["USD"], "fx_rate": [5.0]}).lazy())
    )
    swaps = cast(
        LazyFrame[SwapInput],
        pl.DataFrame(
            {
                "swap_id": ["SWP-FX"],
                "notional": [1_000_000.0],
                "id_indexador_ativo": [1],
                "id_indexador_passivo": [2],
                "indexador_kind_ativo": ["CDI"],
                "indexador_kind_passivo": ["FIXED"],
                "payment_days": [252],
                "accrual": [1.0],
                "base_date": [date(2026, 6, 3)],
                "fixed_rate_ativo": [None],
                "fixed_rate_passivo": [0.08],
                "real_coupon_ativo": [None],
                "real_coupon_passivo": [None],
                "currency_ativo": ["USD"],  # receive CDI in USD, converted at 5.0
                "currency_passivo": ["BRL"],  # no BRL quote -> stays local (1.0)
            }
        ).lazy(),
    )

    result = cast(pl.DataFrame, price_swap(swaps, market).collect())

    # ativo: receive CDI 1e6 * 0.12 * 1, discounted at 0.10, then * 5.0 (USD).
    local_ativo = 1_000_000.0 * 0.12 * math.exp(-0.10)
    assert result.select("ativo_pv").item() == pytest.approx(local_ativo * 5.0, rel=1e-6)
    # passivo: pay fixed 1e6 * 0.08 * 1, discounted at 0.05, local (no BRL quote).
    local_passivo = -1_000_000.0 * 0.08 * math.exp(-0.05)
    assert result.select("passivo_pv").item() == pytest.approx(local_passivo, rel=1e-6)


def test_swap_without_currency_is_unchanged(swap_inputs, swap_market) -> None:
    """No currency columns and no fx_rates source: the optional FX join is
    skipped and PVs are identical to the single-currency baseline."""
    result = cast(pl.DataFrame, price_swap(swap_inputs, swap_market).collect())

    assert result.select("ativo_pv").item() == pytest.approx(108_580.490164, rel=1e-6)
    assert result.select("passivo_pv").item() == pytest.approx(-77_239.829269, rel=1e-6)
