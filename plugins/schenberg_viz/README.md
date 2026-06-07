# schenberg-viz

Visualization and debug helpers for Schenberg FormulaGraph, Router, Fold, and
`schenberg_distributed.ValuationPlan` objects. The plugin works by duck typing:
objects with `to_mermaid`, `info`, or `explain` can be rendered without the core
knowing about this plugin.

`to_mermaid`, `to_markdown`, `to_html`, and `graph_report` never collect data.
`stage_preview` is an explicit debug helper and intentionally collects a small
limited preview.

```python
from schenberg.pricing import forward_formula
from schenberg_viz import to_html, to_markdown

html = to_html(forward_formula, view="output")
markdown = to_markdown(forward_formula, view="output")
```

For valuation plans:

```python
from schenberg_distributed import ValuationPlan
from schenberg_viz import to_mermaid

plan = ValuationPlan("book")
print(to_mermaid(plan))
```
