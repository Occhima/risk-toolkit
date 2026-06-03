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


class OptionGreeks(DataFrameModel):
    """Risk output contract: the five sensitivities attached to a priced option.

    Field order is the canonical Greek order shared with the computation layer
    (:data:`schenberg.math.black_scholes.GREEK_NAMES`)."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class OptionMarketInput(DataFrameModel):
    """Option trade after graph-declared market data has been attached."""

    option_id: str
    instrument_type: str
    option_model: str
    option_kind: str
    id_indexador: int
    spot: float
    strike: float
    payment_days: int
    vol: float


class OptionPricedState(DataFrameModel):
    """Internal option state consumed by numeric/autodiff Greek engines."""

    option_id: str
    instrument_type: str
    option_model: str
    option_kind: str
    id_indexador: int
    spot: float
    strike: float
    payment_days: int
    vol: float
    rate: float
    cost_of_carry: float
    year_fraction: float
    d1: float
    d2: float
    price: float


class OptionPrice(DataFrameModel):
    """Public option price output."""

    option_id: str
    instrument_type: str
    price: float


class OptionPriceWithGreeks(DataFrameModel):
    """Public option price output with Greeks."""

    option_id: str
    instrument_type: str
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
