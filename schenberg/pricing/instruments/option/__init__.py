"""Vanilla European options under generalized Black-Scholes-Merton."""

from __future__ import annotations

from schenberg.pricing.instruments.option.models import option_router
from schenberg.pricing.instruments.option.prices import (
    price_options,
    price_options_with_greeks,
)

__all__ = ["option_router", "price_options", "price_options_with_greeks"]
