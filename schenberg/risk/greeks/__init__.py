"""Greeks for generalized BSM options, three ways, plus a lazy Polars bridge.

* :func:`compute_greeks` — numpy in, dict-of-arrays out, dispatched over
  ``analytic`` / ``numeric`` / ``autodiff``. All three share one price model
  (:mod:`.model`) so they reconcile.
* :func:`attach_greeks` — a vectorized (``map_batches``, no row-wise UDF) step
  that adds ``delta, gamma, vega, theta, rho`` columns to a priced lazy frame.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray

from schenberg.domain.enums import GreekMethod, OptionKind
from schenberg.risk.greeks.analytic import greeks_analytic
from schenberg.risk.greeks.autodiff import greeks_autodiff
from schenberg.risk.greeks.model import generalized_price
from schenberg.risk.greeks.numeric import greeks_numeric

__all__ = [
    "GREEK_NAMES",
    "attach_greeks",
    "compute_greeks",
    "generalized_price",
]

GREEK_NAMES = ("delta", "gamma", "vega", "theta", "rho")

_ENGINES = {
    GreekMethod.ANALYTIC: greeks_analytic,
    GreekMethod.NUMERIC: greeks_numeric,
    GreekMethod.AUTODIFF: greeks_autodiff,
}


def compute_greeks(
    *,
    method: GreekMethod | str,
    spot,
    strike,
    rate,
    carry,
    vol,
    ttm,
    eta,
) -> dict[str, NDArray[np.float64]]:
    """Dispatch to one of the three engines. ``eta`` is +1 call / -1 put."""
    return _ENGINES[GreekMethod(method)](spot, strike, rate, carry, vol, ttm, eta)


def _eta_expr(kind_col: str) -> pl.Expr:
    return pl.when(pl.col(kind_col) == OptionKind.CALL).then(1.0).otherwise(-1.0)


def attach_greeks(
    lf: pl.LazyFrame,
    *,
    method: GreekMethod | str = GreekMethod.ANALYTIC,
    kind_col: str = "option_kind",
    carry_col: str = "cost_of_carry",
    ttm_col: str = "ttm",
) -> pl.LazyFrame:
    """Add the five Greek columns to a priced frame — stays lazy.

    Expects the inputs the pricing graph already surfaced: ``spot``, ``strike``,
    ``rate``, the cost of carry, ``vol`` and a time-to-maturity column.
    """
    fields = ["spot", "strike", "rate", carry_col, "vol", ttm_col, "_eta"]
    struct_dtype = pl.Struct({name: pl.Float64 for name in GREEK_NAMES})

    def run(s: pl.Series) -> pl.Series:
        col = {f: s.struct.field(f).to_numpy() for f in fields}
        greeks = compute_greeks(
            method=method,
            spot=col["spot"],
            strike=col["strike"],
            rate=col["rate"],
            carry=col[carry_col],
            vol=col["vol"],
            ttm=col[ttm_col],
            eta=col["_eta"],
        )
        return pl.DataFrame({name: greeks[name] for name in GREEK_NAMES}).to_struct()

    return (
        lf.with_columns(_eta_expr(kind_col).alias("_eta"))
        .with_columns(
            pl.struct(fields).map_batches(run, return_dtype=struct_dtype).alias("_greeks")
        )
        .drop("_eta")
        .unnest("_greeks")
    )
