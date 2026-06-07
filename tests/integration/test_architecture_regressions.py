from __future__ import annotations

from pathlib import Path

from tests.integration.option_pricer import vanilla_option_graph


def test_pure_option_graph_does_not_require_side() -> None:
    info = vanilla_option_graph.info(view="output")
    forbidden = {"side", "pay_receive", "long_short", "book", "position_id"}

    assert forbidden.isdisjoint(info.required_inputs)
    assert forbidden.isdisjoint(info.view_nodes)


def test_docs_examples_command_is_consistent() -> None:
    docs = [Path("README.md"), Path("docs/concepts.md"), Path("docs/extending.md")]

    for path in docs:
        text = path.read_text()
        assert "uv run poe examples-html" in text
        assert "docs/examples/*.qmd" in text
        assert "marimo export html" not in text
