"""Mermaid helpers for Schenberg introspectable objects."""

from __future__ import annotations

from typing import Any


def to_mermaid(obj: Any, **kwargs: Any) -> str:
    if hasattr(obj, "to_mermaid"):
        return obj.to_mermaid(**kwargs)
    raise TypeError(f"object of type {type(obj).__name__!r} does not provide to_mermaid()")
