from __future__ import annotations

import pytest

from schenberg_distributed import ValuationPlan


def test_node_decorator_registers_node() -> None:
    plan = ValuationPlan("p").input("x", 1)

    @plan.node("y", x="x")
    def y(x: int) -> int:
        return x + 1

    assert plan.has("y")
    assert plan.depends_on("y") == ("x",)
    assert plan.nodes() == ("y",)


def test_node_decorator_returns_original_function() -> None:
    plan = ValuationPlan("p").input("x", 1)

    @plan.node("y", x="x")
    def y(x: int) -> int:
        return x + 1

    assert y(2) == 3


def test_implicit_node_name() -> None:
    plan = ValuationPlan("p").input("x", 1)

    @plan.node(x="x")
    def my_node(x: int) -> int:
        return x + 1

    assert plan.has("my_node")
    assert plan.nodes() == ("my_node",)


def test_duplicate_input_or_node_fails() -> None:
    plan = ValuationPlan("p").input("x", 1)
    with pytest.raises(ValueError, match="already has"):
        plan.input("x", 2)

    @plan.node("y", x="x")
    def y(x: int) -> int:
        return x

    with pytest.raises(ValueError, match="already has"):

        @plan.node("y", x="x")
        def y_again(x: int) -> int:
            return x


def test_missing_dependency_fails() -> None:
    plan = ValuationPlan("p")
    with pytest.raises(ValueError, match="missing dependency"):

        @plan.node("y", x="x")
        def y(x: int) -> int:
            return x


def test_topological_order_with_rustworkx() -> None:
    plan = ValuationPlan("p").input("a", 1)

    @plan.node("b", a="a")
    def b(a: int) -> int:
        return a + 1

    @plan.node("c", b="b")
    def c(b: int) -> int:
        return b + 1

    assert plan.topological_order("c") == ("a", "b", "c")


def test_affected_by_market_source_includes_downstream() -> None:
    plan = ValuationPlan("p").input("trades", 1).input("market", 2)

    @plan.node("price", trades="trades", market="market", market_sources=("curves",))
    def price(trades: int, market: int) -> int:
        return trades + market

    @plan.node("rollup", price="price")
    def rollup(price: int) -> int:
        return price

    assert plan.affected_by_market_source("curves") == ("price", "rollup")
