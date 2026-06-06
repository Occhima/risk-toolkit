"""Canonical market-data domain objects.

This subpackage hosts the small, reusable domain layer that turns raw vendor
quotes into canonical :class:`MarketSource` instances ready to live inside a
:class:`MarketSnapshot` and be consumed declaratively by ``MarketRequirements``.

Two market objects are provided:

- :class:`ForwardCurve`: interest-rate / forward curves with calendar,
  day-count, compounding and quote-kind separated into ``CurveConvention``.
- :class:`VolatilitySurface`: volatility matrices / surfaces with axes and
  quote kind separated into ``VolatilityConvention``.

Both objects keep their data as ``polars.LazyFrame`` end to end and never call
``collect`` on the normalization or export paths.
"""

from schenberg.market_data.objects.compounding import Compounding, CompoundingKind
from schenberg.market_data.objects.conventions import (
    Calendar,
    CurveConvention,
    InterpolationKind,
    InterpolationPolicy,
    QuoteKind,
    VolatilityConvention,
    VolQuoteKind,
)
from schenberg.market_data.objects.curves import CurvePoint, ForwardCurve
from schenberg.market_data.objects.errors import MissingMarketSourceError
from schenberg.market_data.objects.volatility import VolatilityPoint, VolatilitySurface

__all__ = [
    "Calendar",
    "Compounding",
    "CompoundingKind",
    "CurveConvention",
    "CurvePoint",
    "ForwardCurve",
    "InterpolationKind",
    "InterpolationPolicy",
    "MissingMarketSourceError",
    "QuoteKind",
    "VolatilityConvention",
    "VolatilityPoint",
    "VolatilitySurface",
    "VolQuoteKind",
]
