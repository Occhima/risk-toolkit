"""Fold: monoidal aggregation of component rows into instrument/portfolio rows.

A :class:`Fold` is the second-lot answer to "group these component rows by a key
and combine their values". It is the one place that ad-hoc ``group_by(...).agg(...)``
blocks collapse into: declare the group keys, the output schema, and one
aggregation per output column, and :meth:`compute` interprets it as a single lazy
Polars ``group_by``. The aggregations are little :class:`Agg` values built by the
:func:`sum_`, :func:`first_`, :func:`count_` and :func:`lit_` helpers, so a fold
is fully inspectable (:meth:`explain`, :meth:`info`, :meth:`to_mermaid`) — not an
opaque expression.

A fold is monoidal: each output is the reduction of a column under an associative
operation (sum, strict_sum, first, count), optionally weighted or filtered, with the empty
group as the unit. The same :class:`Fold` powers both structured-instrument
aggregation and portfolio /
book roll-ups. Nothing here calls ``collect``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

import polars as pl

from schenberg.core.columns import ColumnLike, RoutePredicate, col_name


class AggOp(StrEnum):
    """The associative reduction an :class:`Agg` performs."""

    SUM = "sum"
    STRICT_SUM = "strict_sum"
    FIRST = "first"
    COUNT = "count"
    LIT = "lit"


# A where-clause is either a raw Polars predicate or a schema RoutePredicate
# (``cols(Schema).leg_role == "ativo"``), which carries a readable label.
Where = pl.Expr | RoutePredicate


def _where_expr(where: Where) -> pl.Expr:
    return where.expr() if isinstance(where, RoutePredicate) else where


def _where_label(where: Where) -> str:
    if isinstance(where, RoutePredicate):
        return f"{where.column.name} {where.op} {where.value!r}"
    return str(where)


@dataclass(frozen=True, slots=True)
class Agg:
    """One declarative aggregation: an associative reduction of a column.

    Built by :func:`sum_`, :func:`first_`, :func:`count_`, :func:`lit_` — never
    directly. Carries enough structure to both compile to a ``pl.Expr`` and
    explain itself in words.
    """

    op: AggOp
    column: str | None = None
    weight: str | None = None
    where: Where | None = None
    value: object | None = None

    def to_expr(self) -> pl.Expr:
        """Compile to a Polars aggregation expression (used inside ``.agg``)."""
        if self.op is AggOp.COUNT:
            return pl.len()
        if self.op is AggOp.LIT:
            return pl.lit(self.value)
        assert self.column is not None
        expr = pl.col(self.column)
        if self.weight is not None:
            expr = expr * pl.col(self.weight)
        if self.where is not None:
            expr = expr.filter(_where_expr(self.where))
        match self.op:
            case AggOp.SUM:
                return expr.sum()
            case AggOp.STRICT_SUM:
                return (
                    pl.when(expr.null_count() > 0)
                    .then(pl.lit(None, dtype=pl.Float64))
                    .otherwise(expr.sum())
                )
            case AggOp.FIRST:
                return expr.first()
        raise ValueError(f"unknown aggregation op {self.op!r}")

    def describe(self) -> str:
        """A short human-readable form, e.g. ``sum(weighted_pv where leg_role == 'ativo')``."""
        if self.op is AggOp.COUNT:
            return "count()"
        if self.op is AggOp.LIT:
            return f"lit({self.value!r})"
        assert self.column is not None
        inner = self.column
        if self.weight is not None:
            inner = f"{inner} * {self.weight}"
        if self.where is not None:
            inner = f"{inner} where {_where_label(self.where)}"
        return f"{self.op}({inner})"


def sum_(
    column: ColumnLike, *, weight: ColumnLike | None = None, where: Where | None = None
) -> Agg:
    """Sum a column, optionally weighted by another column and/or filtered."""
    return Agg(
        op=AggOp.SUM,
        column=col_name(column),
        weight=col_name(weight) if weight is not None else None,
        where=where,
    )


def strict_sum_(
    column: ColumnLike, *, weight: ColumnLike | None = None, where: Where | None = None
) -> Agg:
    """Sum a column, returning null if any contributing row is null."""
    return Agg(
        op=AggOp.STRICT_SUM,
        column=col_name(column),
        weight=col_name(weight) if weight is not None else None,
        where=where,
    )


def first_(column: ColumnLike) -> Agg:
    """Take the first value of a column within each group (e.g. carry a key)."""
    return Agg(op=AggOp.FIRST, column=col_name(column))


def count_() -> Agg:
    """Count the rows in each group."""
    return Agg(op=AggOp.COUNT)


def lit_(value: object) -> Agg:
    """Emit a constant column on every group row (e.g. an instrument type tag)."""
    return Agg(op=AggOp.LIT, value=value)


def _as_agg(value: Agg | pl.Expr) -> Agg:
    if isinstance(value, Agg):
        return value
    raise TypeError(
        f"fold aggregation must be an Agg (sum_/first_/count_/lit_), got {value!r}; "
        f"raw pl.Expr aggregations are not accepted so folds stay inspectable"
    )


@dataclass(slots=True)
class Fold:
    """Group component rows by key(s) and reduce each output column with an :class:`Agg`.

    Declare it fluently and interpret it lazily::

        fold = (
            Fold("swap_fold", input_schema=SwapLegStage)
            .by(C.swap_id)
            .returns(
                SwapOutput,
                npv=sum_(C.weighted_pv),
                ativo_pv=sum_(C.weighted_pv, where=C.leg_role == "ativo"),
                passivo_pv=sum_(C.weighted_pv, where=C.leg_role == "passivo"),
            )
        )
        out = fold.compute(component_rows)   # lazy
    """

    name: str
    input_schema: type[Any] | None = None
    group_keys: tuple[str, ...] = ()
    output_schema: object | None = None
    aggregations: dict[str, Agg] = field(default_factory=dict)

    def by(self, *keys: ColumnLike) -> Fold:
        """Set the group-by keys."""
        self.group_keys = tuple(col_name(k) for k in keys)
        return self

    def returns(self, schema: object | None = None, /, **aggregations: Agg | pl.Expr) -> Fold:
        """Declare the output schema and one aggregation per non-key output column.

        Group keys do not need an aggregation — they survive ``group_by`` and are
        re-selected into schema order. With a ``schema`` every non-key field must
        have an aggregation, and no aggregation may name a column outside it.
        """
        aggs = {name: _as_agg(a) for name, a in aggregations.items()}
        if schema is not None:
            fields = list(cast(Any, schema).to_schema().columns.keys())
            keys = set(self.group_keys)
            extra = sorted(set(aggs) - set(fields))
            if extra:
                raise ValueError(f"fold {self.name!r} aggregates columns not in schema: {extra}")
            missing = sorted(set(fields) - keys - set(aggs))
            if missing:
                raise ValueError(
                    f"fold {self.name!r} is missing aggregations for schema fields: {missing}"
                )
        self.output_schema = schema
        self.aggregations = aggs
        return self

    def compute(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        """Interpret the fold as one lazy ``group_by(...).agg(...)``, ordered to
        the output schema when one is declared. Stays lazy."""
        if not self.group_keys:
            raise ValueError(f"fold {self.name!r} has no group keys; call .by(...)")
        exprs = [agg.to_expr().alias(name) for name, agg in self.aggregations.items()]
        grouped = frame.group_by(list(self.group_keys)).agg(exprs)
        if self.output_schema is not None:
            fields = list(cast(Any, self.output_schema).to_schema().columns.keys())
            return grouped.select(fields)
        return grouped

    # ---- introspection ---------------------------------------------------

    def info(self) -> dict[str, object]:
        return {
            "name": self.name,
            "group_keys": list(self.group_keys),
            "schema": getattr(self.output_schema, "__name__", self.output_schema),
            "aggregations": {name: agg.describe() for name, agg in self.aggregations.items()},
        }

    def explain(self) -> str:
        sname = getattr(self.output_schema, "__name__", None)
        lines = [f"Fold {self.name}"]
        lines += ["", "Group by:"]
        lines += [f"  - {key}" for key in self.group_keys] or ["  - (none)"]
        lines += ["", "Aggregations:"]
        lines += [f"  - {name} = {agg.describe()}" for name, agg in self.aggregations.items()]
        if sname:
            lines += ["", "Returns:", f"  - {sname}"]
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        sname = getattr(self.output_schema, "__name__", "output")
        keys = ", ".join(self.group_keys) or "rows"
        lines = ["flowchart LR", f'    components["component rows"] --> group["group by {keys}"]']
        for name, agg in self.aggregations.items():
            safe = agg.describe().replace('"', "'")
            lines.append(f'    group --> {name}["{name} = {safe}"]')
            lines.append(f"    {name} --> output")
        lines.append(f'    output["{sname}"]')
        return "\n".join(lines)
