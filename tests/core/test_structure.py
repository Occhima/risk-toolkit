from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl
import pytest
from schenberg.core.structure import Structure
from schenberg.domain.schemas import SwapLegInput
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.instruments.swap.structure import swap_structure


@pytest.fixture
def market() -> MarketSnapshot:
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
                    {"id_indexador": [2], "tenor_days": [252], "projected_index": [106.0]}
                ).lazy(),
            ),
        ],
    )


def _legs(*, ativo_weight: float = 1.0, passivo_weight: float = -1.0) -> pl.LazyFrame:
    common = {
        "notional": 1_000_000.0,
        "payment_days": 252,
        "accrual": 1.0,
        "base_date": date(2026, 6, 3),
        "fixed_rate": None,
        "cashflow_amount": None,
    }
    return pl.DataFrame(
        [
            {
                "swap_id": "SWP-1",
                "leg_id": "ativo",
                "leg_kind": "CDI",
                "leg_role": "ativo",
                "leg_weight": ativo_weight,
                "id_indexador": 1,
                "real_coupon": None,
                **common,
            },
            {
                "swap_id": "SWP-1",
                "leg_id": "passivo",
                "leg_kind": "IPCA",
                "leg_role": "passivo",
                "leg_weight": passivo_weight,
                "id_indexador": 2,
                "real_coupon": 0.02,
                **common,
            },
        ]
    ).lazy()


def test_components_frame_returns_pure_component_prices(market) -> None:
    out = cast(pl.DataFrame, swap_structure.components_frame(_legs(), market=market).collect())

    # Pure component pricing: no exposure column, both PVs positive (no sign).
    assert "weighted_pv" not in out.columns
    pv = dict(zip(out["leg_id"], out["pv"], strict=True))
    assert pv["ativo"] > 0
    assert pv["passivo"] > 0


def test_stage_returns_prices_weights_and_contributions(market) -> None:
    out = cast(pl.DataFrame, swap_structure.stage(_legs(), market=market).collect())

    assert {"swap_id", "leg_id", "leg_role", "leg_weight", "pv", "weighted_pv"} <= set(out.columns)
    rows = {r["leg_id"]: r for r in out.to_dicts()}
    # weighted_pv = pv * leg_weight — the exposure applied outside the pricing graph.
    assert rows["passivo"]["weighted_pv"] == pytest.approx(-rows["passivo"]["pv"])
    assert rows["ativo"]["weighted_pv"] == pytest.approx(rows["ativo"]["pv"])


def test_compute_folds_to_final_output(market) -> None:
    out = cast(pl.DataFrame, swap_structure.compute(_legs(), market=market).collect())

    assert out.columns == ["swap_id", "npv", "ativo_pv", "passivo_pv"]
    assert out.height == 1
    assert out.select("npv").item() == pytest.approx(31_340.660895, rel=1e-6)
    assert out.select("ativo_pv").item() == pytest.approx(108_580.490164, rel=1e-6)
    assert out.select("passivo_pv").item() == pytest.approx(-77_239.829269, rel=1e-6)


def test_npv_changes_with_leg_weight(market) -> None:
    base = cast(pl.DataFrame, swap_structure.compute(_legs(), market=market).collect())
    flipped = cast(
        pl.DataFrame,
        swap_structure.compute(
            _legs(ativo_weight=1.0, passivo_weight=1.0), market=market
        ).collect(),
    )

    # Same pure pricing, different exposure -> different NPV.
    assert base.select("npv").item() != pytest.approx(flipped.select("npv").item())


def test_exposure_is_applied_outside_the_pricing_graph(market) -> None:
    # The component view (pricing) never sees leg_weight; only stage adds weighting.
    comp = cast(pl.DataFrame, swap_structure.components_frame(_legs(), market=market).collect())
    flipped_comp = cast(
        pl.DataFrame,
        swap_structure.components_frame(_legs(passivo_weight=99.0), market=market).collect(),
    )
    # Pure component PV is invariant to leg_weight.
    assert comp.sort("leg_id")["pv"].to_list() == pytest.approx(
        flipped_comp.sort("leg_id")["pv"].to_list()
    )


def test_explain_describes_components_exposure_and_fold() -> None:
    text = swap_structure.explain()

    assert "Structure swap" in text
    assert "router: swap_leg_router" in text
    assert "view: output" in text
    assert "weighted_pv" in text
    assert "group by: swap_id" in text
    assert "ativo_pv = sum(weighted_pv where leg_role == 'ativo')" in text
    assert "SwapOutput" in text


def test_to_mermaid_shows_macro_pipeline() -> None:
    mermaid = swap_structure.to_mermaid()

    assert "flowchart LR" in mermaid
    assert "swap_leg_router" in mermaid
    assert "exposure" in mermaid
    assert "fold by swap_id" in mermaid


def test_info_summarizes_structure() -> None:
    info = swap_structure.info()

    assert info["name"] == "swap"
    assert info["input"] == "SwapLegInput"
    assert info["component"] == "swap_leg_router"
    assert info["exposure"] == ["weighted_pv"]
    assert cast(dict, info["fold"])["group_keys"] == ["swap_id"]


def test_diagnose_is_clean_for_a_well_formed_structure() -> None:
    report = swap_structure.diagnose()
    assert report.ok
    assert not report.has_errors


def test_diagnose_reports_an_incomplete_structure() -> None:
    broken = Structure("broken", input=SwapLegInput)  # no components, exposure, or fold
    report = broken.diagnose()

    assert report.has_errors
    codes = {d.code for d in report.diagnostics}
    assert {"no-components", "no-fold"} <= codes
