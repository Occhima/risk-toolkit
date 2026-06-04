"""The single keyed-join builder behind every plain market spec.

``CurveSpec``, ``FxRatesSpec``, ``FixingsSpec`` and the DI curve used to each
hand-roll the same ``MarketRequirement(table=..., on=ColumnSet.from_pairs(...),
outputs={value: output})`` block. That construction now lives here once;
the named specs are thin, readable wrappers that pin a table and the semantic
key names (``indexer``, ``tenor``, ``currency``, ``date``) and delegate the
actual build to :meth:`JoinSpec.read`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import overload

from schenberg.core.columns import ColumnBinding, ColumnLike, ColumnSet, col_name
from schenberg.core.market import MarketRead, MarketRequirement


@dataclass(frozen=True, slots=True)
class JoinSpec:
    """A market table addressed by a keyed left join.

    ``read`` builds a single-value market dependency; each ``key`` is a
    ``(trade_side_column, quote_side_column)`` pair. With ``output`` omitted it
    returns a delayed :class:`MarketRead`, whose output column ``g.market`` names
    from its keyword; with ``output`` given it returns the concrete
    :class:`MarketRequirement`.
    """

    table: str

    @overload
    def read(
        self, value_col: str, *keys: tuple[ColumnLike, str], output: None = ...
    ) -> MarketRead: ...

    @overload
    def read(
        self, value_col: str, *keys: tuple[ColumnLike, str], output: str
    ) -> MarketRequirement: ...

    def read(
        self,
        value_col: str,
        *keys: tuple[ColumnLike, str],
        output: str | None = None,
    ) -> MarketRequirement | MarketRead:
        on = ColumnSet(tuple(ColumnBinding(col_name(left), right) for left, right in keys))

        def build(out: str) -> MarketRequirement:
            return MarketRequirement(table=self.table, on=on, outputs={value_col: out})

        if output is None:
            return MarketRead(build=build)
        return build(output)
