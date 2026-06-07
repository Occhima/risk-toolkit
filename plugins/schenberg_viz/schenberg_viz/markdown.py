"""Markdown reports for Schenberg graphs and plans."""

from __future__ import annotations

from pprint import pformat
from typing import Any

from .mermaid import to_mermaid


def _call_optional(method: Any, **kwargs: Any) -> Any:
    try:
        return method(**kwargs)
    except TypeError:
        return method()


def to_markdown(obj: Any, title: str | None = None, **kwargs: Any) -> str:
    heading = title or getattr(obj, "name", type(obj).__name__)
    parts = [f"# {heading}", "", "```mermaid", to_mermaid(obj, **kwargs), "```"]
    if hasattr(obj, "info"):
        parts += ["", "## Info", "", "```text", pformat(_call_optional(obj.info, **kwargs)), "```"]
    if hasattr(obj, "explain"):
        parts += [
            "",
            "## Explain",
            "",
            "```text",
            str(_call_optional(obj.explain, **kwargs)),
            "```",
        ]
    return "\n".join(parts) + "\n"
