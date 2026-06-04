from __future__ import annotations

from typing import cast

import polars as pl
from schenberg.core.pipeline import Workflow


def _workflow() -> Workflow:
    wf = Workflow("portfolio")

    @wf.stage
    def normalized(raw):  # noqa: ANN001
        return raw

    @wf.stage
    def priced(normalized, market):  # noqa: ANN001
        return normalized.with_columns(price=pl.lit(1.0) + pl.lit(market))

    return wf


def test_run_executes_stages_lazily() -> None:
    wf = _workflow()
    env = wf.run(raw=pl.DataFrame({"x": [1]}).lazy(), market=2.0)

    assert {"raw", "market", "normalized", "priced"}.issubset(env)
    assert isinstance(env["priced"], pl.LazyFrame)
    expected_price = 1.0 + 2.0
    assert env["priced"].collect().select("price").item() == expected_price


def test_order_is_topological() -> None:
    order = _workflow().order()
    assert order.index("normalized") < order.index("priced")


def test_stages_and_external_inputs() -> None:
    wf = _workflow()
    assert set(wf.stages()) == {"normalized", "priced"}
    assert set(wf.external_inputs()) == {"raw", "market"}


def test_info_and_explain_and_mermaid() -> None:
    wf = _workflow()

    info = wf.info()
    assert info["stages"] == ["normalized", "priced"]
    assert cast(dict, info["dependencies"])["priced"] == ["normalized", "market"]

    text = wf.explain()
    assert "Workflow portfolio" in text
    assert "normalized -> priced" in text

    mermaid = wf.to_mermaid()
    assert "flowchart LR" in mermaid
    assert "normalized --> priced" in mermaid
