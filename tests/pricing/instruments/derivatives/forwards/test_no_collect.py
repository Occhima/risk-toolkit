from __future__ import annotations

from pathlib import Path


def test_forward_runtime_files_do_not_call_collect() -> None:
    root = Path("schenberg/pricing/instruments/derivatives/forwards")
    files = [path for path in root.rglob("*.py") if not path.name.startswith("test_")]

    assert files, "no runtime files found"

    for path in files:
        text = path.read_text()
        assert ".collect(" not in text, f"{path}: .collect() call found"
