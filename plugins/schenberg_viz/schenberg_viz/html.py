"""Self-contained HTML output for Schenberg diagrams."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from .markdown import to_markdown
from .mermaid import to_mermaid
from .png import mermaid_png_url


def _call_optional(method: Any, **kwargs: Any) -> Any:
    try:
        return method(**kwargs)
    except TypeError:
        return method()


def _formulas_html(obj: Any) -> str:
    if not hasattr(obj, "formulas"):
        return ""
    formulas = obj.formulas()
    items = "\n".join(
        f"      <li><code>{html.escape(name)}</code>: "
        f'<span class="math">\\({html.escape(formula)}\\)</span></li>'
        for name, formula in formulas.items()
    )
    return "\n".join(["    <h2>Formulas</h2>", '    <ul class="formulas">', items, "    </ul>"])


def to_html(obj: Any, title: str | None = None, **kwargs: Any) -> str:
    """Return an HTML report that renders diagrams and math visibly.

    The report includes both a Mermaid live block and a PNG fallback endpoint so
    exported notebooks are still useful in viewers that do not execute Mermaid.
    """
    heading = title or getattr(obj, "name", type(obj).__name__)
    mermaid = to_mermaid(obj, **kwargs)
    markdown = to_markdown(obj, title=heading, **kwargs)
    png_url = mermaid_png_url(mermaid)
    explain = ""
    if hasattr(obj, "explain"):
        explain = str(_call_optional(obj.explain, **kwargs))
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8">',
            f"  <title>{html.escape(heading)}</title>",
            "  <style>",
            "    body{font-family:Inter,system-ui,sans-serif;margin:2rem;",
            "      background:#0f172a;color:#e2e8f0}",
            "    .card{background:#111827;border:1px solid #334155;border-radius:16px;",
            "      padding:1rem;margin:1rem 0}",
            "    img{max-width:100%;background:white;border-radius:12px;padding:12px}",
            "    pre{white-space:pre-wrap;background:#020617;border-radius:12px;",
            "      padding:1rem;overflow:auto}",
            "    code{color:#67e8f9}.math{font-size:1.05rem}",
            "  </style>",
            "</head>",
            "<body>",
            f"  <h1>{html.escape(heading)}</h1>",
            '  <section class="card">',
            "    <h2>Graph PNG</h2>",
            f'    <img src="{html.escape(png_url)}" alt="{html.escape(heading)} graph">',
            "  </section>",
            '  <section class="card">',
            "    <h2>Live Mermaid</h2>",
            f'    <pre class="mermaid">{html.escape(mermaid)}</pre>',
            "  </section>",
            '  <section class="card">',
            _formulas_html(obj),
            "  </section>",
            '  <section class="card">',
            "    <h2>Explain</h2>",
            f"    <pre>{html.escape(explain or markdown)}</pre>",
            "  </section>",
            '  <script type="module" src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs"></script>',
            '  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>',
            '  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/'
            'auto-render.min.js" onload="renderMathInElement(document.body);"></script>',
            "</body>",
            "</html>",
        ]
    )


def write_html(obj: Any, path: str | Path, title: str | None = None, **kwargs: Any) -> Path:
    output = Path(path)
    output.write_text(to_html(obj, title=title, **kwargs), encoding="utf-8")
    return output
