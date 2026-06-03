from __future__ import annotations

from schenberg.domain.base import DataFrameModel


class OptionTrade(DataFrameModel):
    """A vanilla European option priced under generalized Black-Scholes-Merton.

    ``option_model`` selects how the cost of carry is formed (see
    :class:`~schenberg.domain.enums.OptionModel`); ``id_indexador`` is the
    underlying used to join curves, dividends and the volatility surface.
    """

    option_id: str
    instrument_type: str
    option_model: str
    option_kind: str
    id_indexador: int
    spot: float
    strike: float
    payment_days: int


class OptionPricing(DataFrameModel):
    """Pricing output: the price plus the standardized moneyness terms."""

    d1: float
    d2: float
    price: float
