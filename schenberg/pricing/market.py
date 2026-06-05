"""The market-source registry: fluent specs that pricing requirements read from.

Each singleton names one market table and exposes its readable quantities as
methods returning a :class:`~schenberg.market_data.requirements.Keyed` builder.
The builder already knows its join keys and their *default* contract columns, so
a requirements field is usually just ``requires(CURVES.zero_rate())`` and grows a
``.by(...)`` only when a contract names its columns unconventionally::

    CURVES.zero_rate().by(curve=contract.discount_curve, tenor=contract.payment_date)

Adding a source is a few lines: pin its table name and declare each read's
``Key``\\ s (semantic name, quote-side column, default contract column).
"""

from __future__ import annotations

from dataclasses import dataclass

from schenberg.market_data.requirements import Key, Keyed


@dataclass(frozen=True, slots=True)
class CurveSource:
    """Zero/forward rates off a curve table, keyed by curve name and tenor."""

    name: str = "curves"

    def zero_rate(self) -> Keyed:
        return self._rate("zero_rate")

    def forward_rate(self) -> Keyed:
        return self._rate("forward_rate")

    def _rate(self, value_col: str) -> Keyed:
        return Keyed(
            table=self.name,
            value_col=value_col,
            keys=(
                Key("curve", quote_col="curve_name", default="discount_curve"),
                Key("tenor", quote_col="tenor", default="payment_date"),
            ),
        )


@dataclass(frozen=True, slots=True)
class EnergyForwardSource:
    """Energy forward prices, keyed by submarket, delivery period and fixing date."""

    name: str = "energy_forward_curve"

    def price(self) -> Keyed:
        return Keyed(
            table=self.name,
            value_col="forward_price",
            keys=(
                Key("submarket", quote_col="submarket", default="submarket"),
                Key("delivery_period", quote_col="delivery_period", default="delivery_period"),
                Key("fixing_date", quote_col="fixing_date", default="energy_fixing_date"),
            ),
        )


@dataclass(frozen=True, slots=True)
class FxSource:
    """Spot FX, keyed by the ordered currency pair and a fixing date."""

    name: str = "fx_rates"

    def spot(self) -> Keyed:
        return Keyed(
            table=self.name,
            value_col="fx_rate",
            keys=(
                Key("from_ccy", quote_col="from_ccy", default="ccy"),
                Key("to_ccy", quote_col="to_ccy", default="base_ccy"),
                Key("fixing_date", quote_col="fixing_date", default="energy_fixing_date"),
            ),
        )


@dataclass(frozen=True, slots=True)
class InflationSource:
    """Projected inflation forward factors, keyed by index and fixing date."""

    name: str = "inflation"

    def forward_factor(self) -> Keyed:
        return Keyed(
            table=self.name,
            value_col="forward_factor",
            keys=(
                Key("indexador", quote_col="indexador", default="indexador"),
                Key("fixing_date", quote_col="fixing_date", default="inflation_fixing_date"),
            ),
        )


CURVES = CurveSource("curves")
ENERGY_FWD = EnergyForwardSource("energy_forward_curve")
FX = FxSource("fx_rates")
INFLATION = InflationSource("inflation")

__all__ = [
    "CURVES",
    "ENERGY_FWD",
    "FX",
    "INFLATION",
    "CurveSource",
    "EnergyForwardSource",
    "FxSource",
    "InflationSource",
]
