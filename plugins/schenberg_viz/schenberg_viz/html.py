"""Self-contained HTML output for Schenberg diagrams."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from .markdown import to_markdown
from .mermaid import to_mermaid


def to_html(obj: Any, title: str | None = None, **kwargs: Any) -> str:
    heading = title or getattr(obj, "name", type(obj).__name__)
    mermaid = to_mermaid(obj, **kwargs)
    markdown = to_markdown(obj, title=heading, **kwargs)
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8">',
            f"  <title>{html.escape(heading)}</title>",
            "</head>",
            "<body>",
            f"  <h1>{html.escape(heading)}</h1>",
            f'  <pre class="mermaid">{html.escape(mermaid)}</pre>',
            f"  <pre>{html.escape(markdown)}</pre>",
            '  <script type="module" src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs"></script>',
            "</body>",
            "</html>",
        ]
    )


def write_html(obj: Any, path: str | Path, title: str | None = None, **kwargs: Any) -> Path:
    output = Path(path)
    output.write_text(to_html(obj, title=title, **kwargs), encoding="utf-8")
    return output
