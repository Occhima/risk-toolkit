"""A near-real option book and market for the end-to-end option flow."""

from __future__ import annotations

from datetime import date

import polars as pl
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource

UNDERLYING = 1
SURFACE_TENORS = [63, 126, 252, 504]
SURFACE_STRIKES = [80.0, 90.0, 100.0, 110.0, 120.0]


def _surface_vol(tenor_days: int, strike: float) -> float:
    """A plausible smile (convex in strike) with a gently falling term structure."""
    moneyness = strike / 100.0 - 1.0
    smile = 0.20 + 0.6 * moneyness**2  # vol rises away from ATM
    term = -0.02 * (tenor_days / 252.0)  # longer tenors a touch calmer
    return round(smile + term, 4)


def make_market(*, zero_rate: float = 0.10, div_yield: float = 0.03) -> MarketSnapshot:
    vol_rows = {
        "id_indexador": [],
        "tenor_days": [],
        "strike": [],
        "implied_vol": [],
    }
    for t in SURFACE_TENORS:
        for k in SURFACE_STRIKES:
            vol_rows["id_indexador"].append(UNDERLYING)
            vol_rows["tenor_days"].append(t)
            vol_rows["strike"].append(k)
            vol_rows["implied_vol"].append(_surface_vol(t, k))

    def curve(col: str, value: float) -> pl.LazyFrame:
        return pl.DataFrame(
            {"id_indexador": [UNDERLYING] * len(SURFACE_TENORS), "tenor_days": SURFACE_TENORS,
             col: [value] * len(SURFACE_TENORS)}
        ).lazy()  # fmt: skip

    return MarketSnapshot.from_sources(
        as_of=date(2026, 6, 3),
        sources=[
            MarketSource("vol_surface", pl.DataFrame(vol_rows).lazy()),
            MarketSource("curves", curve("zero_rate", zero_rate)),
            MarketSource("carry_curve", curve("cost_of_carry", zero_rate)),  # GENERALIZED: b = r
            MarketSource("dividend_curve", curve("div_yield", div_yield)),
        ],
    )


def make_options() -> pl.LazyFrame:
    """24 options: {GENERALIZED, MERTON} x strikes x maturities x {CALL, PUT}."""
    rows: list[dict] = []
    for model in ("GENERALIZED", "MERTON"):
        for strike in (90.0, 100.0, 110.0):
            for tenor in (126, 252):
                for kind in ("CALL", "PUT"):
                    rows.append(
                        {
                            "option_id": f"{model[0]}-{int(strike)}-{tenor}-{kind[0]}",
                            "instrument_type": "OPTION",
                            "option_model": model,
                            "option_kind": kind,
                            "id_indexador": UNDERLYING,
                            "spot": 100.0,
                            "strike": strike,
                            "payment_days": tenor,
                        }
                    )
    return pl.DataFrame(rows).lazy()
