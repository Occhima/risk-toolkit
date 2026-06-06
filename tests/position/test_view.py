"""The position-computation layer: PositionView, measures, and the built-in views."""

from __future__ import annotations

import ast
from typing import cast

import polars as pl
import pytest
import schenberg.position.measures as measures_mod
import schenberg.position.view as view_mod
import schenberg.position.views as views_mod
from schenberg.core.columns import cols
from schenberg.core.fold import Fold
from schenberg.core.graph import uses
from schenberg.domain.schemas.position import (
    BookContract,
    InstrumentPnlExplain,
    InstrumentRisk,
    InstrumentValue,
    Position,
    PositionPnlExplain,
    PositionRisk,
    PositionValue,
    ReportingFx,
)
from schenberg.position import (
    book_value_rollup,
    position_pnl_explain,
    position_risk,
    position_value,
)
from schenberg.position.view import PositionView

PV = cols(PositionValue)

# ---- fixtures ----------------------------------------------------------------


def _positions() -> pl.LazyFrame:
    return Position.from_records(
        [
            {
                "position_id": "P1",
                "book": "B1",
                "instrument_type": "FORWARD",
                "instrument_id": "ENG-1",
                "quantity": 100.0,
                "side": 1.0,
                "unit_notional": 10.0,
            },
            {
                "position_id": "P2",
                "book": "B1",
                "instrument_type": "SWAP",
                "instrument_id": "SWP-1",
                "quantity": 50.0,
                "side": -1.0,
                "unit_notional": 2.0,
            },
        ]
    )


def _values() -> pl.LazyFrame:
    return InstrumentValue.from_records(
        [
            {
                "instrument_type": "FORWARD",
                "instrument_id": "ENG-1",
                "value": 2.0,
                "currency": "USD",
            },  # noqa: E501
            {"instrument_type": "SWAP", "instrument_id": "SWP-1", "value": 4.0, "currency": "BRL"},
        ]
    )


def _book() -> pl.LazyFrame:
    return BookContract.from_records(
        [{"book": "B1", "desk": "Energy", "legal_entity": "LE-1", "reporting_currency": "BRL"}]
    )


def _fx() -> pl.LazyFrame:
    return ReportingFx.from_records(
        [
            {"currency": "USD", "reporting_currency": "BRL", "book_fx": 0.2},
            {"currency": "BRL", "reporting_currency": "BRL", "book_fx": 1.0},
        ]
    )


def _collect(lf: pl.LazyFrame) -> pl.DataFrame:
    return cast(pl.DataFrame, lf.collect())


def _by_id(df: pl.DataFrame, column: str) -> dict[str, float]:
    return dict(zip(df["position_id"].to_list(), df[column].to_list(), strict=True))


# ---- lazy by construction / no collect ---------------------------------------


def test_compute_is_lazy_before_collect() -> None:
    out = position_value(_positions(), value=_values(), book=_book(), fx=_fx())
    assert isinstance(out, pl.LazyFrame)


def test_stage_is_lazy_before_collect() -> None:
    out = position_value.stage(_positions(), value=_values(), book=_book(), fx=_fx())
    assert isinstance(out, pl.LazyFrame)


def test_position_layer_never_calls_collect() -> None:
    for module in (view_mod, measures_mod, views_mod):
        with open(module.__file__, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "collect"  # `.collect()`, not `.collect_schema()`
        ]
        assert not calls, f"{module.__name__} must stay lazy (found a .collect() call)"


# ---- joins -------------------------------------------------------------------


def test_join_position_and_instrument_value() -> None:
    out = _collect(position_value(_positions(), value=_values(), book=_book(), fx=_fx()))
    # mtm = side * quantity * value
    assert _by_id(out, "mtm")["P1"] == pytest.approx(1.0 * 100.0 * 2.0)
    assert _by_id(out, "mtm")["P2"] == pytest.approx(-1.0 * 50.0 * 4.0)


def test_book_metadata_comes_from_book_contract_not_position() -> None:
    # desk / legal_entity / reporting_currency are NOT on Position; they are joined
    # in from BookContract and visible in the staged frame.
    staged = _collect(position_value.stage(_positions(), value=_values(), book=_book(), fx=_fx()))
    assert set(staged["desk"].to_list()) == {"Energy"}
    assert set(staged["reporting_currency"].to_list()) == {"BRL"}
    assert "desk" not in Position.to_schema().columns


def test_reporting_fx_conversion() -> None:
    out = _collect(position_value(_positions(), value=_values(), book=_book(), fx=_fx()))
    # P1 is USD with book_fx 0.2: reported = mtm / 0.2
    assert _by_id(out, "reported_mtm")["P1"] == pytest.approx(200.0 / 0.2)
    # P2 is already BRL with book_fx 1.0
    assert _by_id(out, "reported_mtm")["P2"] == pytest.approx(-200.0 / 1.0)


# ---- measures ----------------------------------------------------------------


def test_exposure_and_notional_measures() -> None:
    out = _collect(position_value(_positions(), value=_values(), book=_book(), fx=_fx()))
    assert _by_id(out, "exposure") == {"P1": pytest.approx(100.0), "P2": pytest.approx(-50.0)}
    assert _by_id(out, "position_notional")["P1"] == pytest.approx(abs(100.0) * 10.0)


def test_position_notional_null_when_unit_notional_missing() -> None:
    positions = Position.from_records(
        [
            {
                "position_id": "P3",
                "book": "B1",
                "instrument_type": "FORWARD",
                "instrument_id": "ENG-1",
                "quantity": 7.0,
                "side": 1.0,
                "unit_notional": None,
            }
        ]
    )
    out = _collect(position_value(positions, value=_values(), book=_book(), fx=_fx()))
    assert out.select("position_notional").item() is None


def test_output_columns_match_schema() -> None:
    out = _collect(position_value(_positions(), value=_values(), book=_book(), fx=_fx()))
    assert out.columns == list(PositionValue.to_schema().columns.keys())


# ---- pnl explain -------------------------------------------------------------


def _pnl_explain() -> pl.LazyFrame:
    return InstrumentPnlExplain.from_records(
        [
            {
                "instrument_type": "FORWARD",
                "instrument_id": "ENG-1",
                "currency": "USD",
                "roll_value_pnl": 1.0,
                "curve_value_pnl": 2.0,
                "fx_value_pnl": 3.0,
                "fixing_value_pnl": 4.0,
                "residual_value_pnl": 0.5,
                "total_value_pnl": 10.5,
            }
        ]
    )


def test_pnl_explain_components_are_lifted_to_the_book() -> None:
    positions = Position.from_records(
        [
            {
                "position_id": "P1",
                "book": "B1",
                "instrument_type": "FORWARD",
                "instrument_id": "ENG-1",
                "quantity": 10.0,
                "side": 1.0,
                "unit_notional": None,
            }
        ]
    )
    out = _collect(position_pnl_explain(positions, pnl=_pnl_explain(), book=_book(), fx=_fx()))
    # exposure * component / book_fx = 10 * curve / 0.2
    assert out.select("curve_mtm_pnl").item() == pytest.approx(10.0 * 2.0 / 0.2)


def test_pnl_explain_total_equals_sum_of_components() -> None:
    positions = Position.from_records(
        [
            {
                "position_id": "P1",
                "book": "B1",
                "instrument_type": "FORWARD",
                "instrument_id": "ENG-1",
                "quantity": 10.0,
                "side": 1.0,
                "unit_notional": None,
            }
        ]
    )
    out = _collect(position_pnl_explain(positions, pnl=_pnl_explain(), book=_book(), fx=_fx()))
    components = ["roll", "curve", "fx", "fixing", "residual"]
    parts = sum(out.select(f"{c}_mtm_pnl").item() for c in components)
    assert out.select("total_mtm_pnl").item() == pytest.approx(parts)
    assert out.columns == list(PositionPnlExplain.to_schema().columns.keys())


# ---- risk factors (the same view shape, lifting Greeks) ----------------------


def _risk() -> pl.LazyFrame:
    return InstrumentRisk.from_records(
        [
            {
                "instrument_type": "OPTION",
                "instrument_id": "OPT-1",
                "currency": "USD",
                "delta": 0.6,
                "gamma": 0.02,
                "vega": 12.0,
                "theta": -3.0,
                "rho": 8.0,
            }
        ]
    )


def test_position_risk_lifts_each_greek_by_exposure() -> None:
    positions = Position.from_records(
        [
            {
                "position_id": "P1",
                "book": "B1",
                "instrument_type": "OPTION",
                "instrument_id": "OPT-1",
                "quantity": 10.0,
                "side": -1.0,  # a short option position
                "unit_notional": None,
            }
        ]
    )
    out = _collect(position_risk(positions, risk=_risk()))
    assert out.columns == list(PositionRisk.to_schema().columns.keys())
    # exposure = side * quantity = -10; position_<greek> = exposure * <greek>
    assert out.select("position_delta").item() == pytest.approx(-10.0 * 0.6)
    assert out.select("position_vega").item() == pytest.approx(-10.0 * 12.0)
    assert out.select("position_theta").item() == pytest.approx(-10.0 * -3.0)


def test_position_risk_is_the_same_view_shape() -> None:
    # value / pnl-explain / risk are all PositionViews — same machinery, different
    # joined quantity; introspection works identically.
    assert "PositionView position_risk -> PositionRisk" in position_risk.explain()
    assert position_risk.info()["measures"][0] == "exposure"
    assert isinstance(position_risk(_positions(), risk=_risk()), pl.LazyFrame)


# ---- null propagation --------------------------------------------------------


def test_missing_instrument_value_propagates_nulls() -> None:
    # ENG-1 has no value row -> the left join yields a null `value`, and every
    # downstream measure is null. stage() pinpoints `value` as the root cause.
    values = InstrumentValue.from_records(
        [{"instrument_type": "SWAP", "instrument_id": "SWP-1", "value": 4.0, "currency": "BRL"}]
    )
    staged = _collect(position_value.stage(_positions(), value=values, book=_book(), fx=_fx()))
    row = staged.filter(pl.col("position_id") == "P1")
    assert row.select("value").item() is None
    assert row.select("mtm").item() is None
    assert row.select("reported_mtm").item() is None
    # the sibling position with a value is unaffected
    sibling = staged.filter(pl.col("position_id") == "P2")
    assert sibling.select("mtm").item() == pytest.approx(-200.0)


# ---- introspection -----------------------------------------------------------


def test_explain_lists_sources_and_measures() -> None:
    text = position_value.explain()
    assert "PositionView position_value -> PositionValue" in text
    assert "value <- InstrumentValue" in text
    assert "exposure" in text and "reported_mtm" in text


def test_info_reports_join_plan_and_measures() -> None:
    info = position_value.info()
    assert info["spine"] == "Position"
    assert info["measures"] == ["exposure", "position_notional", "mtm", "reported_mtm"]
    assert {s["source"] for s in cast(list, info["sources"])} == {"value", "book", "fx"}


def test_to_mermaid_renders_joins_and_measures() -> None:
    diagram = position_value.to_mermaid()
    assert diagram.startswith("flowchart LR")
    assert 'value[("InstrumentValue")]' in diagram
    assert "PositionValue" in diagram


def test_stage_exposes_every_intermediate_measure() -> None:
    staged = position_value.stage(_positions(), value=_values(), book=_book(), fx=_fx())
    names = staged.collect_schema().names()
    for measure in ("exposure", "position_notional", "mtm", "reported_mtm"):
        assert measure in names
    assert "book_fx" in names  # the joined context column is present too


# ---- schema / contract behaviour ---------------------------------------------


def test_compute_output_conforms_to_the_contract_schema() -> None:
    # The boundary validates the output against PositionValue: integer quantities
    # are coerced to the schema's float columns, and the frame is shaped to the
    # contract (exactly the schema columns, in order, with the schema dtypes).
    positions = _positions().with_columns(pl.col("quantity").cast(pl.Int64))
    out = _collect(position_value(positions, value=_values(), book=_book(), fx=_fx()))
    schema = PositionValue.to_schema().columns
    assert out.columns == list(schema)
    assert out.schema["exposure"] == pl.Float64  # coerced from the int quantity


def test_validate_flag_can_be_turned_off() -> None:
    # validate=False skips the boundary contract but stays lazy and equal.
    raw = position_value.compute(
        _positions(), value=_values(), book=_book(), fx=_fx(), validate=False
    )
    assert isinstance(raw, pl.LazyFrame)
    validated = position_value(_positions(), value=_values(), book=_book(), fx=_fx())
    assert (
        _collect(raw).sort("position_id").to_dicts()
        == _collect(validated).sort("position_id").to_dicts()
    )


def test_missing_source_frame_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="missing source 'fx'"):
        position_value.compute(_positions(), value=_values(), book=_book())


def test_non_key_column_collision_is_rejected_at_declaration() -> None:
    class A(Position):
        pass

    class B(BookContract):
        side: float  # collides with Position.side (a non-key column)

    with pytest.raises(ValueError, match="collide"):
        (
            PositionView("collide", output=PositionValue)
            .spine(A)
            .source("a", InstrumentValue, on=("instrument_type", "instrument_id"))
            .source("b", B, on="book")
        )


# ---- reusable-measure parity with the decorator form -------------------------


def test_reusable_measures_match_handwritten_decorator() -> None:
    hand = (
        PositionView("hand", output=PositionValue)
        .spine(Position)
        .source("value", InstrumentValue, on=("instrument_type", "instrument_id"))
        .source("book", BookContract, on="book")
        .source("fx", ReportingFx, on=("currency", "reporting_currency"))
    )
    P, V, FX = hand.position, hand.value, hand.fx

    @hand.measure(symbol="E")
    def exposure(side=uses(P.side), qty=uses(P.quantity)) -> pl.Expr:
        return side * qty

    @hand.measure
    def position_notional(qty=uses(P.quantity), un=uses(P.unit_notional)) -> pl.Expr:
        return qty.abs() * un

    @hand.measure
    def mtm(e=uses(exposure), val=uses(V.value)) -> pl.Expr:
        return e * val

    @hand.measure
    def reported_mtm(m=uses(mtm), fx=uses(FX.book_fx)) -> pl.Expr:
        return m / fx

    hand.returns()

    a = _collect(hand(_positions(), value=_values(), book=_book(), fx=_fx()))
    b = _collect(position_value(_positions(), value=_values(), book=_book(), fx=_fx()))
    assert a.sort("position_id").to_dicts() == b.sort("position_id").to_dicts()


# ---- book roll-up (the layer after the view) ---------------------------------


def test_book_rollup_sums_positions_into_books() -> None:
    valued = position_value(_positions(), value=_values(), book=_book(), fx=_fx())
    rolled = _collect(book_value_rollup.compute(valued))
    assert rolled.height == 1
    assert rolled.filter(pl.col("book") == "B1").select("reported_mtm").item() == pytest.approx(
        1000.0 - 200.0
    )


# ---- by(): aggregation shorthand on PositionView -----------------------------


def test_by_returns_a_fold() -> None:
    rollup = position_value.by(PV.book)
    assert isinstance(rollup, Fold)


def test_by_fold_groups_and_sums_numeric_measures() -> None:
    valued = position_value(_positions(), value=_values(), book=_book(), fx=_fx())
    rolled = _collect(position_value.by(PV.book).compute(valued))
    assert rolled.height == 1
    # P1: reported_mtm = 200/0.2 = 1000; P2: reported_mtm = -200/1.0 = -200
    assert rolled.select("reported_mtm").item() == pytest.approx(1000.0 - 200.0)
    assert rolled.select("mtm").item() == pytest.approx(200.0 - 200.0)
    assert rolled.select("exposure").item() == pytest.approx(100.0 - 50.0)


def test_by_fold_is_lazy() -> None:
    valued = position_value(_positions(), value=_values(), book=_book(), fx=_fx())
    out = position_value.by(PV.book).compute(valued)
    assert isinstance(out, pl.LazyFrame)


def test_by_fold_is_inspectable() -> None:
    rollup = position_value.by(PV.book)
    info = rollup.info()
    assert info["group_keys"] == ["book"]
    assert "mtm" in info["aggregations"]
    assert "reported_mtm" in info["aggregations"]
    # non-numeric columns (position_id) must not appear as aggregations
    assert "position_id" not in info["aggregations"]


def test_by_explain_names_the_fold() -> None:
    text = position_value.by(PV.book).explain()
    assert "position_value_by_book" in text
    assert "book" in text


def test_by_multiple_keys() -> None:
    # two positions in different books
    positions = Position.from_records(
        [
            {
                "position_id": "P1",
                "book": "B1",
                "instrument_type": "FORWARD",
                "instrument_id": "ENG-1",
                "quantity": 100.0,
                "side": 1.0,
                "unit_notional": None,
            },
            {
                "position_id": "P2",
                "book": "B2",
                "instrument_type": "FORWARD",
                "instrument_id": "ENG-1",
                "quantity": 50.0,
                "side": 1.0,
                "unit_notional": None,
            },
        ]
    )
    book2 = BookContract.from_records(
        [
            {"book": "B1", "desk": "Energy", "legal_entity": "LE-1", "reporting_currency": "BRL"},
            {"book": "B2", "desk": "Energy", "legal_entity": "LE-2", "reporting_currency": "BRL"},
        ]
    )
    valued = position_value(positions, value=_values(), book=book2, fx=_fx())
    rolled = _collect(position_value.by(PV.book).compute(valued))
    assert rolled.height == 2


def test_by_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="not columns of PositionValue"):
        position_value.by("nonexistent_col")


def test_by_before_returns_raises() -> None:
    view = PositionView("no_schema")
    with pytest.raises(ValueError, match="no output schema"):
        view.by("book")
