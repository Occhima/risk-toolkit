from __future__ import annotations

from pathlib import Path

ALLOWLIST = {
    Path("schenberg/core/router.py"),  # diagnose() is an explicit diagnostic collector.
    Path("schenberg/market_data/interpolated.py"),  # quote-grid precomputation boundary.
    Path("schenberg/market_data/sources.py"),  # source key validation boundary.
    Path("schenberg/market_data/objects/curves.py"),  # object normalization boundary.
    Path("schenberg/market_data/objects/volatility.py"),  # object normalization boundary.
}


def test_runtime_collect_calls_are_allowlisted() -> None:
    root = Path(__file__).parents[1]
    offenders: list[str] = []
    for path in sorted((root / "schenberg").rglob("*.py")):
        rel = path.relative_to(root)
        if ".collect(" not in path.read_text():
            continue
        if rel not in ALLOWLIST:
            offenders.append(str(rel))
    assert offenders == []
