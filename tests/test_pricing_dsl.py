"""The contract-oriented pricing DSL: requirements resolution + typed graph wiring.

These tests pin the behaviour the requirements DSL promises:

* a ``requires(SPEC.method().by(...))`` field compiles to the engine's keyed
  :class:`MarketRequirement` (the same join object the engine attaches),
* ``.by`` is optional -- a read carries typed default key columns,
* a join key pointed at a non-existent contract column fails at class creation,
* an interpolated read (a vol surface) compiles to an InterpolatedRequirement,
* a :class:`PricingGraph` is a Computation: it publishes typed views, and prices a
  bound book into one lazy plan.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from schenberg.contracts import DataFrameModel, price_function
from schenberg.core.graph import PricingGraph, Term, uses
from schenberg.market_data.requirements import MarketRequirements, contract, requires
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.market import CURVES, DI, FIXINGS, VOL


def test_keyed_requirement_compiles_to_join() -> None:
    class Trade(DataFrameModel):
        id_indexador: int
        payment_days: int

    class Reqs(MarketRequirements[Trade]):
        zero_rate: Term[float] = requires(CURVES.zero_rate())

    dep = Reqs.__requirements__["zero_rate"]
    assert dep.table == "curves"
    assert dep.left_keys == ("id_indexador", "payment_days")  # typed defaults
    assert dep.right_keys == ("id_indexador", "tenor_days")
    assert dep.outputs == {"zero_rate": "zero_rate"}  # field name is the output column


def test_by_overrides_only_the_named_key() -> None:
    class Trade(DataFrameModel):
        id_indexador: int
        settle_days: int

    class Reqs(MarketRequirements[Trade]):
        zero_rate: Term[float] = requires(DI.zero_rate().by(tenor=contract.settle_days))

    dep = Reqs.__requirements__["zero_rate"]
    assert dep.table == "di_curve"
    assert dep.left_keys == ("id_indexador", "settle_days")  # tenor overridden, indexer default


def test_unknown_join_key_rejected_at_declaration() -> None:
    with pytest.raises(ValueError, match="unknown join key"):
        CURVES.zero_rate().by(maturity=contract.payment_days)


def test_bad_contract_column_fails_fast_at_class_creation() -> None:
    class Trade(DataFrameModel):
        id_indexador: int
        payment_days: int

    with pytest.raises(ValueError, match="not a column of the contract schema"):

        class Reqs(MarketRequirements[Trade]):
            base_index: Term[float] = requires(FIXINGS.base_index())  # needs base_date


def test_interpolated_requirement_compiles() -> None:
    class Trade(DataFrameModel):
        id_indexador: int
        payment_days: int
        strike: float

    class Reqs(MarketRequirements[Trade]):
        vol: Term[float] = requires(VOL.implied_vol())

    dep = Reqs.__requirements__["vol"]
    assert dep.table == "vol_surface"
    assert dep.left_keys == ("id_indexador", "payment_days", "strike")
    assert dep.right_keys == ("id_indexador", "tenor_days", "strike")


def test_pricing_graph_is_a_typed_computation() -> None:
    class Trade(DataFrameModel):
        trade_id: str
        notional: float
        id_indexador: int
        payment_days: int

    class Reqs(MarketRequirements[Trade]):
        zero_rate: Term[float] = requires(CURVES.zero_rate())

    class Out(DataFrameModel):
        trade_id: str
        present_value: float

    g = PricingGraph[Trade, Reqs, Out]("toy")
    c, m = g.contract, g.market

    @g.formula
    def year_fraction(d: Term[int] = uses(c.payment_days)) -> pl.Expr:
        return d / 252.0

    @g.formula
    def present_value(
        n: Term[float] = uses(c.notional),
        r: Term[float] = uses(m.zero_rate),
        t: Term[float] = uses(year_fraction),
    ) -> pl.Expr:
        return n * (-r * t).exp()

    g.returns()

    assert g.has_view("output")
    assert g.view_schema("output") is Out

    @price_function
    def price(trades, market):  # noqa: ANN001, ANN202
        return g.plan(g.bind(trades, market=market))

    market = MarketSnapshot.from_sources(
        as_of=date(2026, 6, 5),
        sources=[
            MarketSource(
                "curves",
                pl.DataFrame(
                    {"id_indexador": [1], "tenor_days": [252], "zero_rate": [0.10]}
                ).lazy(),
            )
        ],
    )
    trades = pl.DataFrame(
        {"trade_id": ["T1"], "notional": [100.0], "id_indexador": [1], "payment_days": [252]}
    ).lazy()

    out = price(trades, market).collect()
    assert out.columns == ["trade_id", "present_value"]
    assert out["present_value"][0] == pytest.approx(100.0 * pl.Series([-0.10]).exp()[0])
