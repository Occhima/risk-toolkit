"""Visualization report example.

Run with `uv run marimo edit docs/examples/09_viz_report.py`.
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("# Visualization helpers")
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from schenberg.pricing import forward_formula
    from schenberg_viz import (
        graph_report,
        stage_preview,
        graph_png_url,
        latex_png_url,
        to_html,
        to_markdown,
        to_mermaid,
        write_html,
    )

    return (
        forward_formula,
        graph_png_url,
        graph_report,
        latex_png_url,
        mo,
        pl,
        stage_preview,
        to_html,
        to_markdown,
        to_mermaid,
        write_html,
    )


@app.cell
def _(forward_formula, graph_png_url, to_html, to_markdown, to_mermaid):
    mermaid = to_mermaid(forward_formula, view="output")
    markdown = to_markdown(forward_formula, view="output")
    html = to_html(forward_formula, view="output")
    png_url = graph_png_url(forward_formula, view="output")
    return html, markdown, mermaid, png_url


@app.cell
def _(forward_formula, graph_report):
    report = graph_report(forward_formula, view="output")
    return report


@app.cell
def _(forward_formula, pl, stage_preview):
    frame = pl.DataFrame(
        {
            "instrument_id": ["FWD-1", "FWD-2"],
            "indexer": ["DI", "DI"],
            "currency": ["BRL", "BRL"],
            "strike": [100.0, 101.0],
            "payment_days": [252, 252],
            "forward_rate": [112.0, 113.0],
            "risk_free_rate": [0.10, 0.10],
        }
    ).lazy()
    preview = stage_preview(forward_formula, frame, view="output", n=1)
    return preview


@app.cell(hide_code=True)
def _(forward_formula, latex_png_url, mo, png_url, preview):
    formula_cards = [
        mo.vstack([mo.md(f"**{name}**"), mo.image(latex_png_url(formula), alt=formula)])
        for name, formula in forward_formula.formulas().items()
    ]
    mo.vstack(
        [
            mo.md("## Grafo renderizado por endpoint PNG"),
            mo.image(png_url, alt="Forward pricing graph", width="100%"),
            mo.md("## Fórmulas renderizadas como PNG"),
            mo.vstack(formula_cards),
            mo.md("## Preview"),
            mo.ui.table(preview),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
