"""Curve market specs.

:class:`CurveSpec` is the small, generic curve read used by graphs that declare
their market with ``g.market``: a left join on
``(indexer -> id_indexador, tenor -> tenor_days)`` that pulls one value column.
It is a thin wrapper over the shared :class:`~schenberg.market_data.specs.JoinSpec`
join builder. Specialized curve sources (e.g. the DI curve) live alongside.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast, overload

from schenberg.core.columns import ColumnLike
from schenberg.core.market import MarketRead, MarketRequirement
from schenberg.market_data.specs import JoinSpec

__all__ = ["CurveSpec"]


@dataclass(frozen=True, slots=True)
class CurveSpec:
    """A keyed curve read on ``(indexer -> id_indexador, tenor -> tenor_days)``.

    With ``output`` omitted it returns a delayed :class:`MarketRead` that
    ``g.market(rate=...)`` names from its keyword; with ``output`` given it
    returns the concrete :class:`MarketRequirement`.
    """

    name: str = "curves"

    @overload
    def value(
        self,
        value_col: str,
        *,
        indexer: ColumnLike = ...,
        tenor: ColumnLike = ...,
        output: None = ...,
    ) -> MarketRead: ...

    @overload
    def value(
        self,
        value_col: str,
        *,
        indexer: ColumnLike = ...,
        tenor: ColumnLike = ...,
        output: str,
    ) -> MarketRequirement: ...

    def value(
        self,
        value_col: str,
        *,
        indexer: ColumnLike = "id_indexador",
        tenor: ColumnLike = "payment_days",
        output: str | None = None,
    ) -> MarketRequirement | MarketRead:
        return JoinSpec(self.name).read(
            value_col,
            (indexer, "id_indexador"),
            (tenor, "tenor_days"),
            output=output,
        )

    def zero_rate(self, **kwargs: object) -> MarketRequirement | MarketRead:
        return self.value("zero_rate", **cast(Any, kwargs))

    def forward_rate(self, **kwargs: object) -> MarketRequirement | MarketRead:
        return self.value("forward_rate", **cast(Any, kwargs))
