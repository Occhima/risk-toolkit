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
from pandera.typing.polars import LazyFrame

from schenberg.core.columns import cols
from schenberg.domain.enums import GreeksBackend, OptionKind
from schenberg.domain.schemas.option import OptionGreeks, OptionPricedState
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
STATE = cols(OptionPricedState)
GREEKS = cols(OptionGreeks)
_ETA_COL = "_eta"
_GREEKS_COL = "_greeks"


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

    def attach(self, lf: LazyFrame[OptionPricedState]) -> pl.LazyFrame:
        """Add the :class:`OptionGreeks` columns to a priced option state.

        The contract supplies ``spot``, ``strike``, ``rate``, ``cost_of_carry``,
        ``vol`` and ``year_fraction``. Vectorized via ``map_batches`` — never a
        row-wise UDF.
        """
        fields = [
            STATE.spot.name,
            STATE.strike.name,
            STATE.rate.name,
            STATE.cost_of_carry.name,
            STATE.vol.name,
            STATE.year_fraction.name,
            _ETA_COL,
        ]
        struct_dtype = pl.Struct({name: pl.Float64 for name in _GREEK_COLUMNS})

        def run(s: pl.Series) -> pl.Series:
            col = {f: s.struct.field(f).to_numpy() for f in fields}
            greeks = self.compute(
                spot=col[STATE.spot.name],
                strike=col[STATE.strike.name],
                rate=col[STATE.rate.name],
                carry=col[STATE.cost_of_carry.name],
                vol=col[STATE.vol.name],
                ttm=col[STATE.year_fraction.name],
                eta=col[_ETA_COL],
            )
            return pl.DataFrame({name: greeks[name] for name in _GREEK_COLUMNS}).to_struct()

        result = (
            lf.with_columns(
                pl.when(STATE.option_kind.expr() == OptionKind.CALL.value)
                .then(1.0)
                .otherwise(-1.0)
                .alias(_ETA_COL)
            )
            .with_columns(
                pl.struct(fields).map_batches(run, return_dtype=struct_dtype).alias(_GREEKS_COL)
            )
            .drop(_ETA_COL)
            .unnest(_GREEKS_COL)
        )
        return result
