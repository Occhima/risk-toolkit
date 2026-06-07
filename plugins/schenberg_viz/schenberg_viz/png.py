"""PNG endpoints for Schenberg diagrams.

The helpers in this module intentionally return URLs instead of downloading
images. That keeps notebooks lazy and renderable in static HTML exports: the
browser loads the PNG from a Mermaid rendering endpoint.
"""

from __future__ import annotations

import base64
from typing import Any

from .mermaid import to_mermaid

MERMAID_INK_BASE_URL = "https://mermaid.ink/img"


def mermaid_png_url(mermaid: str, *, base_url: str = MERMAID_INK_BASE_URL) -> str:
    """Return a Mermaid PNG endpoint URL for a diagram definition.

    Mermaid and LaTeX blocks are not guaranteed to render inside every notebook
    Markdown cell or exported HTML target. A PNG endpoint gives marimo an actual
    image to show with ``mo.image(...)``.
    """
    encoded = base64.urlsafe_b64encode(mermaid.encode("utf-8")).decode("ascii")
    return f"{base_url.rstrip('/')}/{encoded}"


def graph_png_url(obj: Any, **kwargs: Any) -> str:
    """Return a PNG endpoint URL for an object that exposes ``to_mermaid``."""
    return mermaid_png_url(to_mermaid(obj, **kwargs))
