"""DV01: the value change of an instrument for a +1bp parallel rate move.

DV01 is a *repricing* sensitivity, not a closed form: it reprices under a shocked
market and differences the result. So it reuses the machinery already in
Schenberg — a pure pricer that emits :class:`InstrumentValue`, and a
:class:`~schenberg.market_data.shocks.Shock` that bumps the rate curve — rather
than re-deriving anything. Nothing here calls ``collect``.

    from schenberg.pricing.api import forward_instrument_value
    from schenberg.risk import Dv01Calculator

    dv01 = Dv01Calculator.parallel(forward_instrument_value)
    sensitivities = dv01.compute(trades, market)   # LazyFrame[InstrumentDv01]
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.core.columns import cols
from schenberg.domain.schemas.position import InstrumentDv01, InstrumentValue
from schenberg.market_data.shocks import Shock, curve_parallel_shift

if TYPE_CHECKING:
    from schenberg.market_data.snapshot import MarketSnapshot

#: A pure pricer: trades + market -> one InstrumentValue row per instrument.
ValueFn = Callable[["pl.LazyFrame", "MarketSnapshot"], "pl.LazyFrame"]

_IV = cols(InstrumentValue)
_BUMPED_VALUE = "_value_bumped"


@dataclass(frozen=True, slots=True)
class Dv01Calculator:
    """Reprice under a rate bump and difference: ``dv01 = V(market + shock) - V(market)``.

    ``value_fn`` is any pure pricer that emits :class:`InstrumentValue` (e.g.
    :func:`~schenberg.pricing.api.forward_instrument_value`); ``shock`` is the
    +1bp bump whose effect DV01 measures. Build the common case — a parallel shift
    of a curve column — with :meth:`parallel`.
    """

    value_fn: ValueFn
    shock: Shock

    @classmethod
    def parallel(
        cls,
        value_fn: ValueFn,
        *,
        source: str = "curves",
        column: str = "risk_free_rate",
        bump: float = 1e-4,
    ) -> Dv01Calculator:
        """A DV01 against a parallel additive shift of one rate column (default
        +1bp on ``curves.risk_free_rate``)."""
        return cls(value_fn, curve_parallel_shift(source=source, column=column, shift=bump))

    def compute(self, trades: pl.LazyFrame, market: MarketSnapshot) -> LazyFrame[InstrumentDv01]:
        """Price the book at the base and shocked markets and difference their
        values per instrument. Stays lazy."""
        base = self.value_fn(trades, market)
        bumped = self.value_fn(trades, market.apply(self.shock))
        keys = [_IV.instrument_type.name, _IV.instrument_id.name]
        joined = base.join(
            bumped.select(*keys, _IV.value.expr().alias(_BUMPED_VALUE)),
            on=keys,
            how="left",
        )
        result = joined.select(
            instrument_type=_IV.instrument_type.expr(),
            instrument_id=_IV.instrument_id.expr(),
            currency=_IV.currency.expr(),
            dv01=pl.col(_BUMPED_VALUE) - _IV.value.expr(),
        )
        return cast("LazyFrame[InstrumentDv01]", result)
