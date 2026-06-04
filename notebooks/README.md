# Notebooks

| File | What |
|------|------|
| [`debugging_capabilities.ipynb`](./debugging_capabilities.ipynb) | A walk through Schenberg's graph-debugging tools on a real CDI swap leg — static introspection, DAG rendering, and `stage()` null-propagation. Fully executed, with inline figures. |
| [`build_debugging_notebook.py`](./build_debugging_notebook.py) | Source-of-truth for the notebook. Re-run to regenerate it (and the PNGs) from a reviewable `.py` file. |
| [`REFACTORING_NOTES.md`](./REFACTORING_NOTES.md) | A pro-quant read on where the library can still be simplified — swaps as routed structured forwards, killing graph-declaration boilerplate, and collapsing the market-data specs. |
| `images/` | Standalone PNGs of the DAG and the null-propagation cone. |

## Regenerate

```bash
uv pip install matplotlib networkx nbformat nbconvert ipykernel   # one-off, notebook-only deps
uv run python notebooks/build_debugging_notebook.py
```

The notebook is generated from the `.py` file so it stays diff-reviewable and
re-runnable; edit the builder, not the `.ipynb`.
