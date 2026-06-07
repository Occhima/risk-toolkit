from __future__ import annotations

import polars as pl
from schenberg.pricing import ForwardInput, forward_formula

from schenberg_distributed import ValuationPlan
from schenberg_viz import graph_report, stage_preview, to_html, to_markdown, to_mermaid


def test_to_mermaid_accepts_forward_formula() -> None:
    assert "flowchart" in to_mermaid(forward_formula, view="output")


def test_to_markdown_contains_mermaid_block() -> None:
    md = to_markdown(forward_formula, view="output")
    assert "```mermaid" in md


def test_to_html_escapes_title() -> None:
    html = to_html(forward_formula, title='<Forward "Plan">', view="output")
    assert "&lt;Forward &quot;Plan&quot;&gt;" in html


def test_graph_report_does_not_collect() -> None:
    class NoCollect:
        name = "x"

        def to_mermaid(self) -> str:
            return "flowchart LR"

        def info(self) -> dict[str, str]:
            return {"x": "y"}

        def explain(self) -> str:
            return "ok"

        def collect(self) -> None:
            raise AssertionError("must not collect")

    report = graph_report(NoCollect())
    assert report["mermaid"] == "flowchart LR"
    assert report["info"] == {"x": "y"}


def test_stage_preview_limits_rows() -> None:
    frame = pl.DataFrame(
        {
            "instrument_id": ["a", "b", "c"],
            "indexer": ["DI", "DI", "DI"],
            "currency": ["BRL", "BRL", "BRL"],
            "strike": [100.0, 101.0, 102.0],
            "payment_days": [252, 252, 252],
            "forward_rate": [110.0, 110.0, 110.0],
            "risk_free_rate": [0.1, 0.1, 0.1],
        }
    ).lazy()
    out = stage_preview(forward_formula, frame, n=2)
    assert out.height == 2
    assert set(ForwardInput.to_schema().columns).issubset(set(frame.collect_schema().names()))


def test_to_mermaid_accepts_valuation_plan() -> None:
    plan = ValuationPlan("p").input("x", 1)
    assert "flowchart" in to_mermaid(plan)
