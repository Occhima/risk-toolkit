"""Price European options under generalized Black-Scholes-Merton, with Greeks.

Run with:  uv run python examples/03_price_an_option.py

The market carries an implied-vol *surface* (quoted on a tenor x strike grid);
the option graph declares volatility as market data and interpolates it before
pricing formulas compile. GENERALIZED takes the cost of carry ``b``
straight from a curve; MERTON derives it as ``b = r - q`` from a dividend curve.
``price_options_with_greeks`` then attaches delta/gamma/vega/theta/rho — here via
autograd, but ``backend="CLOSED_FORM"`` or ``"NUMERIC"`` give the same numbers.
Everything stays lazy until ``.collect()``.
"""

from __future__ import annotations

from datetime import date

import polars as pl
from schenberg.domain.schemas.option import OptionTrade
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.api import price_options_with_greeks

TENORS = [126, 252, 504]
STRIKES = [90.0, 100.0, 110.0]


def _flat_curve(col: str, value: float) -> pl.LazyFrame:
    return pl.DataFrame({"id_indexador": [1] * 3, "tenor_days": TENORS, col: [value] * 3}).lazy()


# --- Market: a vol surface, a discount curve, a carry curve and dividends --------
market = MarketSnapshot.from_sources(
    as_of=date(2026, 6, 3),
    sources=[
        MarketSource(
            "vol_surface",
            pl.DataFrame(
                {
                    "id_indexador": [1] * 9,
                    "tenor_days": [t for t in TENORS for _ in STRIKES],
                    "strike": [k for _ in TENORS for k in STRIKES],
                    "implied_vol": [0.24, 0.21, 0.25, 0.22, 0.20, 0.23, 0.21, 0.19, 0.21],
                }
            ).lazy(),
        ),
        MarketSource("curves", _flat_curve("zero_rate", 0.10)),
        MarketSource("carry_curve", _flat_curve("cost_of_carry", 0.10)),  # GENERALIZED: b = r
        MarketSource("dividend_curve", _flat_curve("div_yield", 0.03)),  # MERTON: b = r - q
    ],
)

# --- A small book: one generalized and one Merton call + put --------------------
options = OptionTrade.from_polars(pl.DataFrame(
    {
        "option_id": ["G-CALL", "G-PUT", "M-CALL", "M-PUT"],
        "instrument_type": ["OPTION"] * 4,
        "option_model": ["GENERALIZED", "GENERALIZED", "MERTON", "MERTON"],
        "option_kind": ["CALL", "PUT", "CALL", "PUT"],
        "id_indexador": [1, 1, 1, 1],
        "spot": [100.0] * 4,
        "strike": [100.0] * 4,
        "payment_days": [252] * 4,
    }
))

result = price_options_with_greeks(options, market, backend="AUTODIFF")
with pl.Config(tbl_width_chars=200, float_precision=4):
    print(result.collect())
