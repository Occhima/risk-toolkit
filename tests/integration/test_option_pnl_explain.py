from __future__ import annotations

import polars as pl
import pytest
from tests.integration import option_pricer as pnl
from tests.integration import option_pricer as vanilla


def test_option_pnl_explain_components_sum() -> None:
    lf = pnl.sample_pnl_explain()
    assert isinstance(lf, pl.LazyFrame)
    out = lf.collect().with_columns(
        (
            pl.col("spot_value_pnl")
            + pl.col("vol_value_pnl")
            + pl.col("rate_value_pnl")
            + pl.col("residual_value_pnl")
        ).alias("sum_components")
    )
    for row in out.to_dicts():
        assert row["sum_components"] == pytest.approx(row["total_value_pnl"], abs=1e-9)
    assert "side" not in vanilla.vanilla_option_graph.required_inputs("output")
