from __future__ import annotations

from datetime import date
from typing import cast

import pandas as pd
import polars as pl
import pytest
from schenberg.core.columns import ColumnRef, cols
from schenberg.core.graph import Formula, FormulaGraph, uses
from schenberg.core.router import Router
from schenberg.domain.schemas.forward import ForwardTrade
from schenberg.domain.schemas.market_data import (
    DiCurveContract,
    FxRatesContract,
)
from schenberg.market_data.path import MarketPath
from schenberg.market_data.shocks import Shock, curve_parallel_shift
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.instruments.derivatives.forwards import forward_formula
from schenberg.pricing.instruments.derivatives.forwards.energy import (
    energy_forward_formula,
)
from schenberg.pricing.market import DI

# Direction (side / pay_receive / long / short / leg_weight) must never enter a
# pure pricing graph; it lives only in a Structure's exposure or the position layer.
FORBIDDEN_IN_PRICING = {"side", "pay_receive", "long", "short", "leg_weight"}


def test_dataframe_model_ergonomic_constructors() -> None:
    records = [{"currency": "BRL", "fx_rate": 1.0}]

    assert (
        cast(pl.DataFrame, FxRatesContract.from_records(records).collect()).select("fx_rate").item()
        == 1.0
    )
    assert (
        cast(
            pl.DataFrame, FxRatesContract.from_vectors(currency=["BRL"], fx_rate=[1.0]).collect()
        ).height
        == 1
    )
    assert (
        cast(pl.DataFrame, FxRatesContract.from_pandas(pd.DataFrame(records)).collect()).height == 1
    )


def test_column_ref_and_schema_cols() -> None:
    F = cols(ForwardTrade)

    assert isinstance(F.instrument_id, ColumnRef)
    assert F.instrument_id.name == "instrument_id"
    assert pl.DataFrame({"instrument_id": ["A"]}).select(F.instrument_id.expr()).item() == "A"
    assert (F.forward_family == "ENERGY").expr().meta.root_names() == ["forward_family"]
    with pytest.raises(AttributeError):
        _ = F.quantity


def test_di_curve_read_keys_on_indexer_and_tenor() -> None:
    req = DI.zero_rate().finalize("zero_rate")
    assert req.table == "di_curve"
    assert req.left_keys == ("id_indexador", "payment_days")
    assert req.right_keys == ("id_indexador", "tenor_days")
    assert req.outputs == {"zero_rate": "zero_rate"}


def test_market_snapshot_from_sources_attach_and_shock() -> None:
    source = MarketSource(
        "di_curve",
        pl.DataFrame(
            {"curve_name": ["DI"], "id_indexador": [1], "tenor_days": [21], "zero_rate": [0.1]}
        ).lazy(),
        DiCurveContract,
    )
    snapshot = MarketSnapshot.from_sources(as_of=date(2026, 6, 3), sources=[source])
    trades = pl.DataFrame({"id_indexador": [1], "payment_days": [21]}).lazy()

    attached = snapshot.attach(trades, DI.zero_rate().finalize("rate"))
    bumped = snapshot.apply(curve_parallel_shift(source="di_curve", shift=0.01))

    assert snapshot.source("di_curve") == source
    expected_rate = 0.1
    assert cast(pl.DataFrame, attached.collect()).select("rate").item() == expected_rate
    # The shock preserves the source schema and does not mutate the original.
    assert bumped.source("di_curve").schema is DiCurveContract
    assert cast(pl.DataFrame, snapshot.source("di_curve").data.collect()).select(
        "zero_rate"
    ).item() == pytest.approx(0.10)
    assert cast(pl.DataFrame, bumped.source("di_curve").data.collect()).select(
        "zero_rate"
    ).item() == pytest.approx(0.11)
    # A MarketPath builds the same shock as a lens-lite modifier.
    via_path = MarketPath("di_curve").column("zero_rate").modify(lambda r: r + 0.01)
    assert isinstance(via_path, Shock)
    assert cast(pl.DataFrame, snapshot.apply(via_path).source("di_curve").data.collect()).select(
        "zero_rate"
    ).item() == pytest.approx(0.11)


def test_router_fallback_receives_all_rows_with_no_cases() -> None:
    graph = FormulaGraph("fallback", input=ForwardTrade)
    t = graph.input

    @graph.formula()
    def priced_marker(payment_days: pl.Expr = uses(t.payment_days)) -> pl.Expr:
        return payment_days * 0 + 1

    graph.returns("pricing", priced_marker=priced_marker)
    router = Router.on(cols(ForwardTrade).forward_family).default(graph)

    out = router.compute(
        pl.DataFrame({"forward_family": ["A"], "payment_days": [2]}).lazy(), view="pricing"
    )

    assert cast(pl.DataFrame, out.collect()).select("priced_marker").item() == 1


@pytest.mark.parametrize("graph", [forward_formula, energy_forward_formula])
def test_side_never_appears_in_pure_pricing_graphs(graph: Formula) -> None:
    """A pure pricing graph returns instrument value, not position value: position
    direction must not be among its inputs or its formula terms."""
    inner = cast(FormulaGraph, graph._g)
    term_names = set(inner._indices)
    assert not (term_names & FORBIDDEN_IN_PRICING), (
        f"pricing graph {graph.name} leaks direction: {sorted(term_names & FORBIDDEN_IN_PRICING)}"
    )
    assert not (graph.required_inputs() & FORBIDDEN_IN_PRICING)
