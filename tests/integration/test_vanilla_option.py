from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import ColumnNotFoundError
from schenberg.market_data.path import MarketPath
from schenberg.risk import Scenario, ScenarioSet, reprice_under
from tests.integration import option_pricer as vanilla


def _priced() -> pl.LazyFrame:
    return vanilla.price_vanilla_option(vanilla.sample_trades(), vanilla.sample_market())


def test_black_scholes_known_values_and_lazy() -> None:
    priced = _priced()
    assert isinstance(priced, pl.LazyFrame)
    out = {r["option_type"]: r for r in priced.collect().to_dicts()}
    call = out["call"]
    put = out["put"]
    assert call["value"] == pytest.approx(10.4506, abs=1e-3)
    assert put["value"] == pytest.approx(5.5735, abs=1e-3)
    assert call["value"] - put["value"] == pytest.approx(
        100 - 100 * 2.718281828459045**-0.05, abs=1e-3
    )
    assert call["delta"] == pytest.approx(0.6368, abs=1e-3)
    assert put["delta"] == pytest.approx(-0.3632, abs=1e-3)
    assert call["gamma"] == pytest.approx(0.01876, abs=1e-4)
    assert call["vega"] == pytest.approx(37.524, abs=1e-2)


def test_option_graph_is_pure_no_side() -> None:
    assert "side" not in vanilla.vanilla_option_graph.required_inputs("output")
    assert "side" not in _priced().collect().columns


def test_price_function_valid_and_invalid_inputs() -> None:
    assert isinstance(
        vanilla.price_vanilla_option(vanilla.sample_trades(), vanilla.sample_market()), pl.LazyFrame
    )
    bad = vanilla.sample_trades().drop("strike")
    with pytest.raises(ColumnNotFoundError):
        vanilla.price_vanilla_option(bad, vanilla.sample_market()).collect()


def test_reprice_under_vol_shock_increases_value() -> None:
    scenarios = ScenarioSet.of(
        Scenario(
            "vol +1pt", MarketPath("vol_surface").column("implied_vol").modify(lambda v: v + 0.01)
        )
    )
    risk = reprice_under(
        vanilla.price_vanilla_option, vanilla.sample_trades(), vanilla.sample_market(), scenarios
    )
    assert isinstance(risk, pl.LazyFrame)
    out = risk.collect()
    assert "scenario" in out.columns
    assert {"value_base", "value_shocked", "value_diff"} <= set(out.columns)
    assert out.filter(pl.col("instrument_id") == "OPT-CALL-1")["value_diff"][0] > 0
