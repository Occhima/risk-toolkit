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

from schenberg.core.columns import ColumnBinding, ColumnLike, ColumnSet, col_name
from schenberg.core.market import MarketRequirement


@dataclass(frozen=True, slots=True)
class JoinSpec:
    """A market table addressed by a keyed left join.

    ``read`` builds a single-value :class:`MarketRequirement`; each ``key`` is a
    ``(trade_side_column, quote_side_column)`` pair. With ``output`` omitted the
    requirement's output defaults to ``value_col`` and is renamed later by
    ``FormulaGraph.for_market``.
    """

    table: str

    def read(
        self,
        value_col: str,
        *keys: tuple[ColumnLike, str],
        output: str | None = None,
    ) -> MarketRequirement:
        on = ColumnSet(tuple(ColumnBinding(col_name(left), right) for left, right in keys))
        return MarketRequirement(table=self.table, on=on, outputs={value_col: output or value_col})
