from __future__ import annotations

from collections.abc import Iterable


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
            f"market snapshot has no source {name!r}; "
            f"available sources: {list(self.available)}"
        )
        super().__init__(message)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.args[0] if self.args else ""
