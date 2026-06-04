"""The swap leg graphs price *pure components* — no position direction anywhere.

These tests guard the most important correction of the second refactor: side /
pay_receive / ativo / passivo / leg_weight must never appear inside a pricing
graph. They live in the swap :class:`Structure`'s exposure/fold instead.
"""

from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl
import pytest
from pandera.typing.polars import LazyFrame
from schenberg.domain.schemas import LegPricing, SwapLegInput
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.instruments.swap.generic import base_swap_leg_graph
from schenberg.pricing.instruments.swap.legs.fixed import fixed_swap_leg_graph
from schenberg.pricing.instruments.swap.pricing import price_swaps
from schenberg.pricing.instruments.swap.structure import swap_structure


def test_leg_pricing_schema_has_no_signed_cashflow() -> None:
    fields = set(LegPricing.to_schema().columns.keys())
    assert "signed_cashflow" not in fields
    assert fields == {"year_fraction", "discount_factor", "cashflow_amount", "pv"}


def test_leg_graphs_contain_no_direction_formulas() -> None:
    for graph in (base_swap_leg_graph, fixed_swap_leg_graph):
        terms = set(graph.topological_order())
        assert "pay_receive_sign" not in terms
        assert "signed_cashflow" not in terms
        # direction inputs are never read by the pricing graph
        assert "pay_receive" not in graph.required_inputs()
        assert "leg_weight" not in graph.required_inputs()
        assert "side" not in graph.required_inputs()


def _market() -> MarketSnapshot:
    # The swap Structure runs the full leg router, so every branch's market read
    # must resolve even when no row routes to it. A full DI/IPCA market provides
    # forward_rate / fixings / projected_indexes columns alongside zero_rate.
    return MarketSnapshot.from_sources(
        as_of=date(2026, 6, 3),
        sources=[
            MarketSource(
                "curves",
                pl.DataFrame(
                    {
                        "id_indexador": [1],
                        "tenor_days": [252],
                        "zero_rate": [0.10],
                        "forward_rate": [0.12],
                    }
                ).lazy(),
            ),
            MarketSource(
                "fixings",
                pl.DataFrame(
                    {
                        "id_indexador": [1],
                        "fixing_date": [date(2026, 6, 3)],
                        "fixing_value": [100.0],
                    }
                ).lazy(),
            ),
            MarketSource(
                "projected_indexes",
                pl.DataFrame(
                    {"id_indexador": [1], "tenor_days": [252], "projected_index": [106.0]}
                ).lazy(),
            ),
        ],
    )


def _fixed_leg(*, leg_weight: float, leg_role: str) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "swap_id": ["S"],
            "leg_id": ["fixed"],
            "leg_kind": ["FIXED"],
            "leg_role": [leg_role],
            "leg_weight": [leg_weight],
            "notional": [1_000_000.0],
            "id_indexador": [1],
            "payment_days": [252],
            "accrual": [1.0],
            "base_date": [date(2026, 6, 3)],
            "fixed_rate": [0.08],
            "real_coupon": [None],
            "cashflow_amount": [None],
        }
    ).lazy()


def test_pure_leg_pv_is_invariant_to_leg_weight() -> None:
    market = _market()
    receive = cast(
        pl.DataFrame,
        fixed_swap_leg_graph.compute(
            _fixed_leg(leg_weight=1.0, leg_role="ativo"), market=market, view="pricing"
        ).collect(),
    )
    pay = cast(
        pl.DataFrame,
        fixed_swap_leg_graph.compute(
            _fixed_leg(leg_weight=-1.0, leg_role="passivo"), market=market, view="pricing"
        ).collect(),
    )

    # Pure pricing does not change when the position direction changes.
    assert receive.select("pv").item() == pytest.approx(pay.select("pv").item())
    assert receive.select("pv").item() > 0


def test_structure_npv_changes_sign_with_leg_weight() -> None:
    market = _market()
    receive = cast(
        pl.DataFrame,
        swap_structure.compute(
            _fixed_leg(leg_weight=1.0, leg_role="ativo"), market=market
        ).collect(),
    )
    pay = cast(
        pl.DataFrame,
        swap_structure.compute(
            _fixed_leg(leg_weight=-1.0, leg_role="passivo"), market=market
        ).collect(),
    )

    assert receive.select("npv").item() == pytest.approx(-pay.select("npv").item())


def test_price_swaps_delegates_to_structure() -> None:
    market = _market()
    legs = cast(LazyFrame[SwapLegInput], _fixed_leg(leg_weight=1.0, leg_role="ativo"))

    via_facade = cast(pl.DataFrame, price_swaps(legs, market).collect())
    via_structure = cast(pl.DataFrame, swap_structure.compute(legs, market=market).collect())

    assert via_facade.equals(via_structure)
