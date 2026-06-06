from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import polars as pl


class MissingMarketSourceError(KeyError, ValueError):
    """Raised when a market snapshot is queried for an unknown source name.

    Inherits from both :class:`KeyError` (idiomatic mapping miss) and
    :class:`ValueError` (backward compatibility with earlier snapshot behaviour
    which raised plain ``ValueError``).
    """

    def __init__(self, name: str, available: Iterable[str] = ()) -> None:
        self.name = name
        self.available = tuple(sorted(available))
        message = (
            f"market snapshot has no source {name!r}; available sources: {list(self.available)}"
        )
        super().__init__(message)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.args[0] if self.args else ""


class DuplicateMarketKeyError(ValueError):
    """Raised when a market source contains duplicate quote keys."""

    def __init__(self, source: str, keys: Iterable[str], duplicates: pl.DataFrame) -> None:
        self.source = source
        self.keys = tuple(keys)
        self.duplicates = duplicates
        count = duplicates.height
        rows = cast(Any, duplicates.to_dicts())
        super().__init__(
            f"market source {source!r} has duplicate quote keys {list(self.keys)}; "
            f"duplicate rows/count: {count}; rows: {rows}"
        )
