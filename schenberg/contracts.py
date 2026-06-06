"""Public contract surface: dataframe models and the pricer entry-point decorator.

``SchenbergDataFrameModel`` is the Pandera base every boundary schema extends.
``price_function`` marks a top-level pricer and validates its typed LazyFrame
inputs/outputs lazily, so contract violations surface at ``collect`` with a
clear schema error rather than as a downstream column-not-found.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandera.polars as pa

from schenberg.domain.base import SchenbergDataFrameModel

__all__ = ["SchenbergDataFrameModel", "price_function"]


def price_function[F: Callable[..., Any]](fn: F) -> F:
    """Validate a pricer's annotated ``LazyFrame[...]`` inputs and output lazily."""
    return pa.check_types(lazy=True)(fn)
