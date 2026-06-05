"""The contract-oriented pricing DSL: requirements resolution and end-to-end pricing.

These tests pin the behaviour the requirements DSL promises:

* a ``requires(SPEC.method().by(...))`` field compiles to the engine's existing
  keyed :class:`MarketRequirement` (same join object ``FormulaGraph.market`` uses),
* ``.by`` is optional — a read carries typed default key columns,
* a join key pointed at a non-existent contract column fails at class creation,
* the whole instrument prices to one lazy plan with the expected numbers.
"""

from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest
from pandera.typing.polars import LazyFrame
from schenberg.contracts import DataFrameModel, price_function
from schenberg.core.graph import PricingGraph, Term, uses
from schenberg.market_data.requirements import MarketRequirements, contract, requires
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.market import CURVES, ENERGY_FWD, FX, INFLATION

INFL_FIX, ENERGY_FIX, PAY = date(2026, 6, 30), date(2026, 8, 6), date(2027, 7, 15)


class EnergyForwardContract(DataFrameModel):
    trade_id: str
    indexador: str
    submarket: str
    delivery_period: str
    inflation_fixing_date: date
    energy_fixing_date: date
    payment_date: date
    payment_days: int
    quantity_mwh: float
    strike: float
    base_index: float
    ccy: str
    base_ccy: str
    discount_curve: str


class EnergyForwardRequirements(MarketRequirements[EnergyForwardContract]):
    projected_index: Term[float] = requires(
        INFLATION.forward_factor().by(
            indexador=contract.indexador,
            fixing_date=contract.inflation_fixing_date,
        )
    )
    energy_forward_price: Term[float] = requires(
        ENERGY_FWD.price().by(
            submarket=contract.submarket,
            delivery_period=contract.delivery_period,
            fixing_date=contract.energy_fixing_date,
        )
    )
    zero_rate: Term[float] = requires(
        CURVES.zero_rate().by(
            curve=contract.discount_curve,
            tenor=contract.payment_date,
        )
    )
    fx_rate: Term[float] = requires(
        FX.spot().by(
            from_ccy=contract.ccy,
            to_ccy=contract.base_ccy,
            fixing_date=contract.energy_fixing_date,
        )
    )


class EnergyForwardOutput(DataFrameModel):
    trade_id: str
    future_value: float
    present_value: float
    value: float


def _graph() -> PricingGraph:
    g = PricingGraph[
        EnergyForwardContract, EnergyForwardRequirements, EnergyForwardOutput
    ]("energy_forward_ipca")
    c, m = g.contract, g.market

    @g.formula
    def year_fraction(days: Term[int] = uses(c.payment_days)) -> pl.Expr:
        return days / 252.0

    @g.formula
    def inflation_factor(
        projected: Term[float] = uses(m.projected_index),
        base: Term[float] = uses(c.base_index),
    ) -> pl.Expr:
        return projected / base

    @g.formula
    def real_spread(
        fwd: Term[float] = uses(m.energy_forward_price),
        strike: Term[float] = uses(c.strike),
    ) -> pl.Expr:
        return fwd - strike

    @g.formula
    def future_value(
        quantity: Term[float] = uses(c.quantity_mwh),
        spread: Term[float] = uses(real_spread),
        inflation: Term[float] = uses(inflation_factor),
    ) -> pl.Expr:
        return quantity * spread * inflation

    @g.formula
    def discount_factor(
        r: Term[float] = uses(m.zero_rate), t: Term[float] = uses(year_fraction)
    ) -> pl.Expr:
        return (-r * t).exp()

    @g.formula
    def present_value(
        fv: Term[float] = uses(future_value), df: Term[float] = uses(discount_factor)
    ) -> pl.Expr:
        return fv * df

    @g.formula
    def value(
        pv: Term[float] = uses(present_value), fx: Term[float] = uses(m.fx_rate)
    ) -> pl.Expr:
        return pv * fx

    g.returns()
    return g


def _market() -> MarketSnapshot:
    return MarketSnapshot.from_sources(
        as_of=date(2026, 6, 5),
        sources=[
            MarketSource(
                "inflation",
                pl.DataFrame(
                    {"indexador": ["IPCA"], "fixing_date": [INFL_FIX], "forward_factor": [110.0]}
                ).lazy(),
            ),
            MarketSource(
                "energy_forward_curve",
                pl.DataFrame(
                    {
                        "submarket": ["SE"],
                        "delivery_period": ["2026-07"],
                        "fixing_date": [ENERGY_FIX],
                        "forward_price": [120.0],
                    }
                ).lazy(),
            ),
            MarketSource(
                "curves",
                pl.DataFrame({"curve_name": ["DI"], "tenor": [PAY], "zero_rate": [0.10]}).lazy(),
            ),
            MarketSource(
                "fx_rates",
                pl.DataFrame(
                    {
                        "from_ccy": ["USD"],
                        "to_ccy": ["BRL"],
                        "fixing_date": [ENERGY_FIX],
                        "fx_rate": [5.0],
                    }
                ).lazy(),
            ),
        ],
    )


def _trades() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "trade_id": ["ENG-1"],
            "indexador": ["IPCA"],
            "submarket": ["SE"],
            "delivery_period": ["2026-07"],
            "inflation_fixing_date": [INFL_FIX],
            "energy_fixing_date": [ENERGY_FIX],
            "payment_date": [PAY],
            "payment_days": [252],
            "quantity_mwh": [10.0],
            "strike": [100.0],
            "base_index": [100.0],
            "ccy": ["USD"],
            "base_ccy": ["BRL"],
            "discount_curve": ["DI"],
        }
    ).lazy()


def test_requirements_compile_to_keyed_joins() -> None:
    deps = EnergyForwardRequirements.__requirements__
    assert set(deps) == {"projected_index", "energy_forward_price", "zero_rate", "fx_rate"}

    zero = deps["zero_rate"]
    assert zero.table == "curves"
    # left = contract columns named by .by(), right = quote-side join columns.
    assert zero.left_keys == ("discount_curve", "payment_date")
    assert zero.right_keys == ("curve_name", "tenor")
    assert zero.outputs == {"zero_rate": "zero_rate"}  # field name is the output column

    fx = deps["fx_rate"]
    assert fx.left_keys == ("ccy", "base_ccy", "energy_fixing_date")
    assert fx.right_keys == ("from_ccy", "to_ccy", "fixing_date")


def test_by_is_optional_typed_defaults() -> None:
    """A contract whose columns match a read's default keys needs no ``.by``."""

    class DefaultsContract(DataFrameModel):
        indexador: str
        inflation_fixing_date: date

    class DefaultsReqs(MarketRequirements[DefaultsContract]):
        projected_index: Term[float] = requires(INFLATION.forward_factor())

    dep = DefaultsReqs.__requirements__["projected_index"]
    assert dep.left_keys == ("indexador", "inflation_fixing_date")
    assert dep.right_keys == ("indexador", "fixing_date")


def test_unknown_join_key_rejected_at_declaration() -> None:
    with pytest.raises(ValueError, match="unknown join key"):
        CURVES.zero_rate().by(maturity=contract.payment_date)


def test_bad_contract_column_fails_fast_at_class_creation() -> None:
    with pytest.raises(ValueError, match="not a column of the contract schema"):

        class BadReqs(MarketRequirements[EnergyForwardContract]):
            zero_rate: Term[float] = requires(
                CURVES.zero_rate().by(tenor=contract.payment_dayz)  # typo
            )


def test_prices_energy_forward_end_to_end() -> None:
    g = _graph()

    @price_function
    def price(
        trades: LazyFrame[EnergyForwardContract], market: MarketSnapshot
    ) -> LazyFrame[EnergyForwardOutput]:
        return g.plan(g.bind(trades, market=market))

    out = price(_trades(), _market()).collect()  # type: ignore[arg-type]
    row = out.to_dicts()[0]

    assert out.columns == ["trade_id", "future_value", "present_value", "value"]
    assert row["trade_id"] == "ENG-1"
    # quantity * (fwd - strike) * (projected / base) = 10 * 20 * 1.1
    assert row["future_value"] == pytest.approx(220.0)
    # future_value * exp(-zero_rate * year_fraction), year_fraction = 252/252 = 1
    assert row["present_value"] == pytest.approx(220.0 * math.exp(-0.10))
    # present_value * fx_rate
    assert row["value"] == pytest.approx(220.0 * math.exp(-0.10) * 5.0)


def test_plan_stays_lazy_until_collect() -> None:
    g = _graph()
    plan = g.plan(g.bind(_trades(), market=_market()))
    assert isinstance(plan, pl.LazyFrame)
