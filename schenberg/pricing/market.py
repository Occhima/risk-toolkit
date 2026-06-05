"""The market-source registry: fluent specs that pricing requirements read from.

Each singleton names one market table and exposes its readable quantities as
methods returning a fluent read (:class:`~schenberg.market_data.requirements.Keyed`
for an exact join, :class:`~schenberg.market_data.requirements.Interpolated` for a
grid). Every read already knows its join keys and their *default* contract
columns, so a requirements field is usually just ``requires(CURVES.zero_rate())``
and grows a ``.by(...)`` only when a contract names its columns unconventionally.

Conventions follow the toolkit's market schemas: curves key on
``(id_indexador -> id_indexador, payment_days -> tenor_days)``; fixings on
``(id_indexador, fixing_date)``; the vol surface interpolates ``implied_vol`` over
``(tenor_days, strike)`` grouped by ``id_indexador``; energy forwards key on
``(submarket, delivery_period)``; FX on ``currency``.
"""

from __future__ import annotations

from dataclasses import dataclass

from schenberg.market_data.requirements import Interpolated, Key, Keyed

# Standard curve keys: indexer + tenor, defaulting to the trade's payment horizon.
_INDEXER = Key("indexer", quote_col="id_indexador", default="id_indexador")
_TENOR = Key("tenor", quote_col="tenor_days", default="payment_days")


def _curve(table: str, value_col: str) -> Keyed:
    return Keyed(table=table, value_col=value_col, keys=(_INDEXER, _TENOR))


@dataclass(frozen=True, slots=True)
class CurveSource:
    """A keyed curve read on ``(id_indexador, tenor_days)`` pulling one value."""

    name: str = "curves"

    def value(self, value_col: str) -> Keyed:
        return _curve(self.name, value_col)

    def zero_rate(self) -> Keyed:
        return self.value("zero_rate")

    def forward_rate(self) -> Keyed:
        return self.value("forward_rate")

    def cost_of_carry(self) -> Keyed:
        return self.value("cost_of_carry")

    def div_yield(self) -> Keyed:
        return self.value("div_yield")

    def projected_index(self) -> Keyed:
        return self.value("projected_index")


@dataclass(frozen=True, slots=True)
class FixingsSource:
    """A fixing value keyed by index and date (``base_date`` for inflation bases)."""

    name: str = "fixings"

    def base_index(self) -> Keyed:
        return Keyed(
            table=self.name,
            value_col="fixing_value",
            keys=(_INDEXER, Key("date", quote_col="fixing_date", default="base_date")),
        )

    def value(self) -> Keyed:
        return Keyed(
            table=self.name,
            value_col="fixing_value",
            keys=(_INDEXER, Key("date", quote_col="fixing_date", default="fixing_date")),
        )


@dataclass(frozen=True, slots=True)
class VolSurfaceSource:
    """An implied-vol surface, grouped by ``id_indexador`` and bilinearly
    interpolated over ``(tenor_days, strike)``."""

    name: str = "vol_surface"

    def implied_vol(self) -> Interpolated:
        return Interpolated(
            table=self.name,
            value_col="implied_vol",
            group=_INDEXER,
            axes=(
                Key("tenor", quote_col="tenor_days", default="payment_days"),
                Key("strike", quote_col="strike", default="strike"),
            ),
        )


@dataclass(frozen=True, slots=True)
class EnergyForwardSource:
    """Energy forward prices keyed by submarket and delivery period."""

    name: str = "energy_forward_curve"

    def price(self) -> Keyed:
        return Keyed(
            table=self.name,
            value_col="forward_price",
            keys=(
                Key("submarket", quote_col="submarket", default="submarket"),
                Key("period", quote_col="delivery_period", default="delivery_period"),
            ),
        )


@dataclass(frozen=True, slots=True)
class FxSource:
    """Spot FX keyed by currency."""

    name: str = "fx_rates"

    def rate(self) -> Keyed:
        return Keyed(
            table=self.name,
            value_col="fx_rate",
            keys=(Key("currency", quote_col="currency", default="currency"),),
        )


CURVES = CurveSource("curves")
DI = CurveSource("di_curve")
CARRY = CurveSource("carry_curve")
DIVIDENDS = CurveSource("dividend_curve")
PROJECTED = CurveSource("projected_indexes")
FIXINGS = FixingsSource("fixings")
VOL = VolSurfaceSource("vol_surface")
ENERGY_FWD = EnergyForwardSource("energy_forward_curve")
FX = FxSource("fx_rates")

__all__ = [
    "CARRY",
    "CURVES",
    "DI",
    "DIVIDENDS",
    "ENERGY_FWD",
    "FIXINGS",
    "FX",
    "PROJECTED",
    "VOL",
    "CurveSource",
    "EnergyForwardSource",
    "FixingsSource",
    "FxSource",
    "VolSurfaceSource",
]
