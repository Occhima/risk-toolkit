from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass(frozen=True, slots=True)
class MarketSource:
    name: str
    data: pl.LazyFrame
    schema: type[Any] | None = None
