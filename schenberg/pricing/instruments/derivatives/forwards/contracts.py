from __future__ import annotations

from datetime import date

import pandera.polars as pa

from schenberg.domain.base import SchenbergDataFrameModel
from schenberg.domain.rules import rule_for
from schenberg.market_data import date_rules as dates


class IndexerFixingMixin(SchenbergDataFrameModel):
    """Contract mixin that resolves the index fixing coordinate."""

    indexer: str
    index_fixing_date: date | None = pa.Field(nullable=True)

    @rule_for("index_fixing_date", selector="indexer", value="CPI")
    def _cpi(cls):  # noqa: N805
        return dates.add_days("tenor", 5)

    @rule_for("index_fixing_date", selector="indexer", default=True)
    def _default(cls):  # noqa: N805
        return dates.same_day("tenor")


class CurrencyFixingMixin(SchenbergDataFrameModel):
    """Contract mixin that resolves the currency fixing coordinate."""

    currency: str
    currency_fixing_date: date | None = pa.Field(nullable=True)

    @rule_for("currency_fixing_date", selector="currency", value="EUR")
    def _eur(cls):  # noqa: N805
        return dates.previous_business_day("tenor")

    @rule_for("currency_fixing_date", selector="currency", default=True)
    def _default(cls):  # noqa: N805
        return dates.same_day("tenor")


class TenorMixin(SchenbergDataFrameModel):
    """Contract mixin for forward tenor/payment horizon."""

    tenor: date
    payment_days: int


class ForwardContractPricing(
    IndexerFixingMixin,
    CurrencyFixingMixin,
    TenorMixin,
    SchenbergDataFrameModel,
):
    """Normalized contract schema for forward-like instruments.

    Specialized forwards may extend this class with additional contract columns,
    but should reuse the same forward formula when the payoff remains:

        future_value = forward_price - strike
        present_value = future_value * discount_factor
        value = present_value * currency
    """

    instrument_id: str
    strike: float


class ForwardPricing(SchenbergDataFrameModel):
    """Output schema for forward pricing."""

    future_value: float
    present_value: float
    value: float
