"""Option Greeks — three backends behind one contract.

* ``CLOSED_FORM`` is graph-native: :data:`bsm_greeks_graph` composes onto the
  option pricing graph and the Greeks fall out as ordinary Polars expressions
  (no Python callback). This is what the option pricer uses by default.
* ``NUMERIC`` and ``AUTODIFF`` cannot be Polars expressions (they bump / reverse-
  mode differentiate the numpy price model), so :class:`GreeksEngine` bridges the
  numpy kernels in :mod:`schenberg.math.black_scholes` onto a lazy frame via a
  single vectorized ``map_batches``.

All three emit the :class:`OptionGreeks` columns and reconcile, because they
share one price model.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from schenberg.domain.enums import GreeksBackend, OptionKind
from schenberg.domain.schemas.option import OptionGreeks
from schenberg.math.black_scholes import (
    GREEK_NAMES,
    Greeks,
    greeks_analytic,
    greeks_autodiff,
    greeks_numeric,
)
from schenberg.risk.greeks.graph import bsm_greeks_graph

__all__ = [
    "GREEK_NAMES",
    "GreeksBackend",
    "GreeksEngine",
    "bsm_greeks_graph",
]

_KERNELS = {
    GreeksBackend.CLOSED_FORM: greeks_analytic,
    GreeksBackend.NUMERIC: greeks_numeric,
    GreeksBackend.AUTODIFF: greeks_autodiff,
}

# Output columns driven by the contract, so frame and schema cannot drift.
_GREEK_COLUMNS = tuple(OptionGreeks.to_schema().columns.keys())


@dataclass(frozen=True, slots=True)
class GreeksEngine:
    """The numpy Greek engine: choose a backend once, reuse it.

    Serves the ``NUMERIC`` / ``AUTODIFF`` backends on lazy frames and the
    one-shot numpy ``compute`` (the reconciliation reference for all three).
    """

    backend: GreeksBackend = GreeksBackend.CLOSED_FORM

    def compute(self, *, spot, strike, rate, carry, vol, ttm, eta) -> Greeks:
        """Run the chosen kernel: numpy in, dict-of-arrays out. ``eta`` is +1/-1."""
        return _KERNELS[self.backend](spot, strike, rate, carry, vol, ttm, eta)

    def attach(
        self,
        lf: pl.LazyFrame,
        *,
        kind_col: str = "option_kind",
        carry_col: str = "cost_of_carry",
        ttm_col: str = "ttm",
    ) -> pl.LazyFrame:
        """Add the :class:`OptionGreeks` columns to a priced frame — stays lazy.

        Expects the columns the pricing graph already surfaced: ``spot``,
        ``strike``, ``rate``, the cost of carry, ``vol`` and a time-to-maturity.
        Vectorized via ``map_batches`` — never a row-wise UDF.
        """
        fields = ["spot", "strike", "rate", carry_col, "vol", ttm_col, "_eta"]
        struct_dtype = pl.Struct({name: pl.Float64 for name in _GREEK_COLUMNS})

        def run(s: pl.Series) -> pl.Series:
            col = {f: s.struct.field(f).to_numpy() for f in fields}
            greeks = self.compute(
                spot=col["spot"],
                strike=col["strike"],
                rate=col["rate"],
                carry=col[carry_col],
                vol=col["vol"],
                ttm=col[ttm_col],
                eta=col["_eta"],
            )
            return pl.DataFrame({name: greeks[name] for name in _GREEK_COLUMNS}).to_struct()

        return (
            lf.with_columns(
                pl.when(pl.col(kind_col) == OptionKind.CALL.value)
                .then(1.0)
                .otherwise(-1.0)
                .alias("_eta")
            )
            .with_columns(
                pl.struct(fields).map_batches(run, return_dtype=struct_dtype).alias("_greeks")
            )
            .drop("_eta")
            .unnest("_greeks")
        )
