default:
    @just --list

sync:
    uv sync --all-groups

format:
    uv run ruff format .

lint:
    uv run ruff check .

typecheck:
    uv run ty check

test:
    uv run pytest

check: lint typecheck test
