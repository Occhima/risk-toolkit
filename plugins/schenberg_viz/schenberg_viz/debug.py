"""Debug and report helpers for Schenberg introspectable objects."""

from __future__ import annotations

from typing import Any

import polars as pl

from .mermaid import to_mermaid


def _call_optional(method: Any, **kwargs: Any) -> Any:
    try:
        return method(**kwargs)
    except TypeError:
        return method()


def graph_report(obj: Any, **kwargs: Any) -> dict[str, Any]:
    report: dict[str, Any] = {}
    if hasattr(obj, "info"):
        report["info"] = _call_optional(obj.info, **kwargs)
    if hasattr(obj, "explain"):
        report["explain"] = _call_optional(obj.explain, **kwargs)
    if hasattr(obj, "to_mermaid"):
        report["mermaid"] = to_mermaid(obj, **kwargs)
    if hasattr(obj, "required_inputs"):
        report["required_inputs"] = _call_optional(obj.required_inputs, **kwargs)
    if hasattr(obj, "topological_order"):
        report["topological_order"] = _call_optional(obj.topological_order, **kwargs)
    return report


def stage_preview(
    graph: Any, frame: pl.LazyFrame, view: str = "output", n: int = 5
) -> pl.DataFrame:
    if not hasattr(graph, "stage"):
        raise TypeError(f"object of type {type(graph).__name__!r} does not provide stage()")
    return graph.stage(frame, view=view).limit(n).collect()
