"""Curve market specs.

:class:`CurveSpec` is the small, generic curve read used by graphs that declare
their market with :meth:`FormulaGraph.for_market`: a left join on
``(indexer -> id_indexador, tenor -> tenor_days)`` that pulls one value column.
It is a thin wrapper over the shared :class:`~schenberg.market_data.specs.JoinSpec`
join builder. Specialized curve sources (e.g. the DI curve) live alongside.
"""

from __future__ import annotations

from dataclasses import dataclass

from schenberg.core.columns import ColumnLike
from schenberg.core.market import MarketRequirement
from schenberg.market_data.specs import JoinSpec

__all__ = ["CurveSpec"]


@dataclass(frozen=True, slots=True)
class CurveSpec:
    """A keyed curve read.

    Returns a :class:`MarketRequirement` whose output defaults to ``value_col``;
    ``FormulaGraph.for_market`` renames it to its keyword, or pass ``output``
    explicitly.
    """

    name: str = "curves"

    def value(
        self,
        value_col: str,
        *,
        indexer: ColumnLike = "id_indexador",
        tenor: ColumnLike = "payment_days",
        output: str | None = None,
    ) -> MarketRequirement:
        return JoinSpec(self.name).read(
            value_col,
            (indexer, "id_indexador"),
            (tenor, "tenor_days"),
            output=output,
        )
