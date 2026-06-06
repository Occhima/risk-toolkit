from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from schenberg.pricing import api


def test_public_pricing_api_is_small_and_working() -> None:
    for name in [
        "price_forward",
        "forward_instrument_value",
        "price_energy_forward",
        "energy_forward_instrument_value",
    ]:
        assert hasattr(api, name)


def test_documented_python_example_imports_exist() -> None:
    root = Path(__file__).parents[2]
    for path in (root / "docs" / "examples").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            spec = importlib.util.find_spec(node.module)
            assert spec is not None, f"{path} imports missing module {node.module}"
            module = __import__(node.module, fromlist=[alias.name for alias in node.names])
            for alias in node.names:
                assert hasattr(module, alias.name), f"{path} imports missing symbol {alias.name}"
