# Agentic engineering notes

This repository is intentionally small and friendly to agent harnesses such as
Codex, Claude, OpenAI API agents, and local automation.

## Expected workflow

1. Read `pyproject.toml` for package metadata, dependency groups, and tooling.
2. Keep pricing formulas in `schenberg/pricing/instruments/**` and reusable graph
   infrastructure in `schenberg/core/**`.
3. Keep Pandera boundary schemas in `schenberg/domain/schemas.py`; avoid leaking
   Pandera into graph-core internals.
4. Run `uv run ruff check .`, `uv run ty check`, and `uv run pytest` before PRs.
5. Prefer lazy Polars operations; do not call `.collect()` inside library code.

## Architecture map

- `schenberg/core/graph.py`: formula DAG and router.
- `schenberg/core/market.py`: lazy market-data joins.
- `schenberg/core/pipeline.py`: topological workflow pipeline.
- `schenberg/domain/schemas.py`: public input/output contracts.
- `schenberg/pricing/api.py`: stable public pricing imports.
- `schenberg/pricing/instruments/swap/engine.py`: swap pricing graphs.
- `schenberg/pricing/instruments/forward/generic.py`: generic forward valuation.
- `schenberg/pricing/instruments/forward/energy.py`: energy forward pricer.
