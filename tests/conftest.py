from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource


@pytest.fixture
def swap_market() -> MarketSnapshot:
    return MarketSnapshot.from_sources(
        as_of=date(2026, 6, 3),
        sources=[
            MarketSource(
                "curves",
                pl.DataFrame(
                    {
                        "id_indexador": [1, 2],
                        "tenor_days": [252, 252],
                        "zero_rate": [0.10, 0.05],
                        "forward_rate": [0.12, None],
                    }
                ).lazy(),
            ),
            MarketSource(
                "fixings",
                pl.DataFrame(
                    {
                        "id_indexador": [2],
                        "fixing_date": [date(2026, 6, 3)],
                        "fixing_value": [100.0],
                    }
                ).lazy(),
            ),
            MarketSource(
                "projected_indexes",
                pl.DataFrame(
                    {
                        "id_indexador": [2],
                        "tenor_days": [252],
                        "projected_index": [106.0],
                    }
                ).lazy(),
            ),
        ],
    )


@pytest.fixture
def swap_inputs() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "swap_id": ["SWP-1"],
            "notional": [1_000_000.0],
            "id_indexador_ativo": [1],
            "id_indexador_passivo": [2],
            "indexador_kind_ativo": ["CDI"],
            "indexador_kind_passivo": ["IPCA"],
            "payment_days": [252],
            "accrual": [1.0],
            "base_date": [date(2026, 6, 3)],
            "fixed_rate_ativo": [None],
            "fixed_rate_passivo": [None],
            "real_coupon_ativo": [None],
            "real_coupon_passivo": [0.02],
        }
    ).lazy()


@pytest.fixture
def energy_market() -> MarketSnapshot:
    return MarketSnapshot.from_sources(
        as_of=date(2026, 6, 3),
        sources=[
            MarketSource(
                "di_curve",
                pl.DataFrame(
                    {
                        "curve_name": ["DI", "DI"],
                        "id_indexador": [1, 1],
                        "tenor_days": [30, 60],
                        "zero_rate": [0.10, 0.10],
                    }
                ).lazy(),
            ),
            MarketSource(
                "energy_forward_curve",
                pl.DataFrame(
                    {
                        "submarket": ["SE", "SE"],
                        "delivery_period": ["2026-07", "2026-08"],
                        "forward_price": [120.0, 130.0],
                        "settle_days": [30, 60],
                    }
                ).lazy(),
            ),
            MarketSource(
                "fx_rates",
                pl.DataFrame({"currency": ["BRL"], "fx_rate": [1.0]}).lazy(),
            ),
        ],
    )


@pytest.fixture
def energy_inputs() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "instrument_id": ["ENG-1", "ENG-1"],
            "instrument_type": ["FORWARD", "FORWARD"],
            "forward_family": ["ENERGY", "ENERGY"],
            "settlement_type": ["PHYSICAL", "PHYSICAL"],
            "submarket": ["SE", "SE"],
            "delivery_period": ["2026-07", "2026-08"],
            "id_indexador": [1, 1],
            "payment_days": [30, 60],
            "strike": [100.0, 100.0],
            "currency": ["BRL", "BRL"],
        }
    ).lazy()
