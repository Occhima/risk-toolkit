from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.instruments.derivatives.forwards.api import price_forward
from schenberg.pricing.instruments.derivatives.forwards.energy.api import (
    energy_forward_formula,
    price_energy_forward,
)
from schenberg.pricing.instruments.derivatives.forwards.energy.contracts import EnergyForwardPricing
from schenberg.pricing.instruments.derivatives.forwards.energy.market import EnergyForwardMarket
from schenberg.pricing.instruments.derivatives.forwards.formulas import (
    build_forward_formula,
    forward_formula,
)
from schenberg.pricing.instruments.derivatives.forwards.market import ForwardMarket


def test_energy_forward_is_built_from_forward_builder() -> None:
    rebuilt = build_forward_formula(
        name="energy_forward", contract=EnergyForwardPricing, market=EnergyForwardMarket
    )
    assert rebuilt.info().formula_nodes == energy_forward_formula.info().formula_nodes
    assert rebuilt.info().formula_nodes == forward_formula.info().formula_nodes


def test_energy_forward_uses_energy_market_without_mutating_base_graph() -> None:
    assert EnergyForwardMarket.__requirements__["forward_price"].table == "energy_forward_curve"
    assert ForwardMarket.__requirements__["forward_price"].table == "curves"
    assert energy_forward_formula.name == "energy_forward"
    assert forward_formula.name == "forward"


def test_forward_and_energy_forward_still_price() -> None:
    market = MarketSnapshot.from_sources(
        as_of=date(2026, 6, 5),
        sources=[
            MarketSource(
                "curves",
                pl.DataFrame(
                    {
                        "id_indexador": ["IDX"],
                        "tenor_days": [252],
                        "forward_rate": [110.0],
                        "risk_free_rate": [0.10],
                    }
                ).lazy(),
                unique_by=("id_indexador", "tenor_days"),
            ),
            MarketSource(
                "energy_forward_curve",
                pl.DataFrame(
                    {
                        "submarket": ["SE"],
                        "delivery_period": ["2026-07"],
                        "forward_price": [270.0],
                    }
                ).lazy(),
                unique_by=("submarket", "delivery_period"),
            ),
        ],
    )
    forward_trades = pl.DataFrame(
        {
            "instrument_id": ["FWD-1"],
            "tenor": [date(2027, 6, 5)],
            "indexer": ["IDX"],
            "currency": ["USD"],
            "strike": [100.0],
            "payment_days": [252],
        }
    ).lazy()
    energy_trades = pl.DataFrame(
        {
            "instrument_id": ["ENG-1"],
            "tenor": [date(2026, 7, 1)],
            "indexer": ["IDX"],
            "currency": ["BRL"],
            "strike": [250.0],
            "payment_days": [252],
            "submarket": ["SE"],
            "incentive": ["I0"],
            "delivery_period": ["2026-07"],
        }
    ).lazy()
    generic = cast(pl.DataFrame, price_forward(forward_trades, market).collect())
    energy = cast(pl.DataFrame, price_energy_forward(energy_trades, market).collect())
    expected_generic_future_value = 10.0
    expected_energy_future_value = 20.0
    assert generic.select("future_value").item() == expected_generic_future_value
    assert energy.select("future_value").item() == expected_energy_future_value
