from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import polars as pl

from schenberg.market_data.objects.errors import DuplicateMarketKeyError


@dataclass(frozen=True, slots=True)
class MarketSource:
    name: str
    data: pl.LazyFrame
    schema: type[Any] | None = None
    unique_by: tuple[str, ...] = ()

    def validate_unique_keys(self) -> None:
        """Validate quote-key uniqueness for this source.

        This may collect the market source metadata/data once at snapshot
        construction. That is an explicit market-data boundary; pricing graph
        construction and trade-side computation remain lazy.
        """
        if not self.unique_by:
            return
        duplicates = cast(
            pl.DataFrame,
            self.data.group_by(list(self.unique_by))
            .len(name="duplicate_count")
            .filter(pl.col("duplicate_count") > 1)
            .collect(),
        )
        if duplicates.is_empty():
            return
        raise DuplicateMarketKeyError(self.name, self.unique_by, duplicates)
