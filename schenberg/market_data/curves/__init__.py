"""Curve market specs.

:class:`CurveSpec` is the small, generic curve read used by graphs that declare
their market with :meth:`FormulaGraph.for_market`: a left join on
``(indexer -> id_indexador, tenor -> tenor_days)`` that pulls one value column.
Specialized curve sources (e.g. the DI curve) live alongside in this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import overload

from schenberg.core.columns import ColumnLike, ColumnSet, col_name
from schenberg.core.market import MarketRead, MarketRequirement

__all__ = ["CurveSpec"]


@dataclass(frozen=True, slots=True)
class CurveSpec:
    """A keyed curve read.

    With ``output`` omitted the spec returns a :class:`MarketRead` whose output
    column is finalized by ``FormulaGraph.for_market``; with ``output`` given it
    returns a concrete :class:`MarketRequirement`.
    """

    name: str = "curves"

    @overload
    def value(
        self,
        value_col: str,
        *,
        indexer: ColumnLike = ...,
        tenor: ColumnLike = ...,
        output: str,
    ) -> MarketRequirement: ...

    @overload
    def value(
        self,
        value_col: str,
        *,
        indexer: ColumnLike = ...,
        tenor: ColumnLike = ...,
        output: None = None,
    ) -> MarketRead: ...

    def value(
        self,
        value_col: str,
        *,
        indexer: ColumnLike = "id_indexador",
        tenor: ColumnLike = "payment_days",
        output: str | None = None,
    ) -> MarketRead | MarketRequirement:
        def build(out: str) -> MarketRequirement:
            return MarketRequirement(
                table=self.name,
                on=ColumnSet.from_pairs(
                    (col_name(indexer), "id_indexador"),
                    (col_name(tenor), "tenor_days"),
                ),
                outputs={value_col: out},
            )

        return build(output) if output is not None else MarketRead(build)
