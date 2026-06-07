"""LaTeX rendering endpoints for notebooks and static docs."""

from __future__ import annotations

from urllib.parse import quote

CODECOGS_PNG_BASE_URL = "https://latex.codecogs.com/png.image"


def latex_png_url(latex: str, *, base_url: str = CODECOGS_PNG_BASE_URL, dpi: int = 140) -> str:
    """Return a PNG endpoint URL for a LaTeX expression."""
    encoded = quote(rf"\dpi{{{dpi}}} {latex}", safe="")
    return f"{base_url}?{encoded}"
