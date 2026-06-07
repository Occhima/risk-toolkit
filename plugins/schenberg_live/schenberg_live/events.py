"""Synchronous in-memory valuation events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True, slots=True)
class MarketEvent:
    source: str
    version: str
    keys: tuple[tuple[str, object], ...] = ()
    as_of: date | None = None
    payload: Any | None = None


@dataclass(frozen=True, slots=True)
class PositionEvent:
    version: str
    book: str | None = None
    instrument_ids: tuple[str, ...] = ()
    payload: Any | None = None
