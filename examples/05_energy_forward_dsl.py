"""The contract-oriented pricing DSL: an IPCA-linked energy forward.

Run with:  uv run python examples/05_energy_forward_dsl.py

This is the recommended shape for declaring an instrument. Read top to bottom:

* a ``Contract`` schema (what each trade row carries),
* a ``Requirements`` schema (what market data the instrument needs, and how to
  find each row) — every join is one line here, and nowhere else,
* the pure formulas (they only ``uses(...)`` contract and market terms),
* an ``Output`` schema and a ``@price_function`` entry point.

The formulas never join; ``g.bind(trades, market)`` resolves the market data and
``g.plan(bound)`` returns one lazy Polars plan. Nothing collects until you ask.
"""

from __future__ import annotations

from datetime import date

import polars as pl
from pandera.typing.polars import LazyFrame
from schenberg.contracts import DataFrameModel, price_function
from schenberg.core.graph import PricingGraph, Term, uses
from schenberg.market_data.requirements import MarketRequirements, contract, requires
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.market import CURVES, ENERGY_FWD, FX, INFLATION


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


g = PricingGraph[
    EnergyForwardContract,
    EnergyForwardRequirements,
    EnergyForwardOutput,
]("energy_forward_ipca")

c = g.contract
m = g.market


@g.formula
def year_fraction(
    days: Term[int] = uses(c.payment_days),
) -> pl.Expr:
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
    r: Term[float] = uses(m.zero_rate),
    T: Term[float] = uses(year_fraction),
) -> pl.Expr:
    return (-r * T).exp()


@g.formula
def present_value(
    fv: Term[float] = uses(future_value),
    df: Term[float] = uses(discount_factor),
) -> pl.Expr:
    return fv * df


@g.formula
def value(
    pv: Term[float] = uses(present_value),
    fx: Term[float] = uses(m.fx_rate),
) -> pl.Expr:
    return pv * fx


g.returns()


@price_function
def price_energy_forwards(
    trades: LazyFrame[EnergyForwardContract],
    market: MarketSnapshot,
) -> LazyFrame[EnergyForwardOutput]:
    bound = g.bind(trades, market=market)
    return g.plan(bound)


def _demo_market() -> MarketSnapshot:
    infl_fix, energy_fix, pay = date(2026, 6, 30), date(2026, 8, 6), date(2027, 7, 15)
    return MarketSnapshot.from_sources(
        as_of=date(2026, 6, 5),
        sources=[
            MarketSource(
                "inflation",
                pl.DataFrame(
                    {"indexador": ["IPCA"], "fixing_date": [infl_fix], "forward_factor": [110.0]}
                ).lazy(),
            ),
            MarketSource(
                "energy_forward_curve",
                pl.DataFrame(
                    {
                        "submarket": ["SE"],
                        "delivery_period": ["2026-07"],
                        "fixing_date": [energy_fix],
                        "forward_price": [120.0],
                    }
                ).lazy(),
            ),
            MarketSource(
                "curves",
                pl.DataFrame(
                    {"curve_name": ["DI"], "tenor": [pay], "zero_rate": [0.10]}
                ).lazy(),
            ),
            MarketSource(
                "fx_rates",
                pl.DataFrame(
                    {
                        "from_ccy": ["USD"],
                        "to_ccy": ["BRL"],
                        "fixing_date": [energy_fix],
                        "fx_rate": [5.0],
                    }
                ).lazy(),
            ),
        ],
    )


if __name__ == "__main__":
    trades = pl.DataFrame(
        {
            "trade_id": ["ENG-1"],
            "indexador": ["IPCA"],
            "submarket": ["SE"],
            "delivery_period": ["2026-07"],
            "inflation_fixing_date": [date(2026, 6, 30)],
            "energy_fixing_date": [date(2026, 8, 6)],
            "payment_date": [date(2027, 7, 15)],
            "payment_days": [252],
            "quantity_mwh": [10.0],
            "strike": [100.0],
            "base_index": [100.0],
            "ccy": ["USD"],
            "base_ccy": ["BRL"],
            "discount_curve": ["DI"],
        }
    ).lazy()

    priced = price_energy_forwards(trades, _demo_market())  # type: ignore[arg-type]
    print(g.explain())
    print()
    print(priced.collect())
