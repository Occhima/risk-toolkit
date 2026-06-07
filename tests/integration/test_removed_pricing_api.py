from __future__ import annotations

from pathlib import Path


def test_pricing_api_module_deleted() -> None:
    assert not Path("schenberg/pricing/api.py").exists()


def test_docs_and_examples_do_not_import_old_pricing_api() -> None:
    roots = [Path("README.md"), Path("docs"), Path("tests")]
    offenders: list[Path] = []
    for root in roots:
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for path in paths:
            if path.suffix not in {".py", ".md", ".html", ".yml", ".toml"}:
                continue
            text = path.read_text(encoding="utf-8")
            old_full = "schenberg.pricing" + ".api"
            old_short = "pricing" + ".api"
            if old_full in text or old_short in text:
                offenders.append(path)
    assert offenders == []


def test_no_new_centralized_example_pricer_module() -> None:
    forbidden = {
        Path("schenberg/pricing/forward.py"),
        Path("schenberg/pricing/forwards.py"),
        Path("schenberg/pricing/pricers.py"),
    }
    assert [path for path in forbidden if path.exists()] == []
