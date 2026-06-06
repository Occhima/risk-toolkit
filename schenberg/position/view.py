"""PositionView: a lazy, typed view whose columns are position *measures*.

A position computation has the same shape as a :class:`~schenberg.core.graph.Formula`,
only the boundary is wider. A formula reads one contract frame plus market reads
attached *before* compilation; a :class:`PositionView` reads a **spine** frame
(the :class:`~schenberg.domain.schemas.position.Position`) plus **context
sources** (the pure :class:`InstrumentValue`, the :class:`BookContract`, the
:class:`ReportingFx`) joined *before* compilation. The measures themselves —
``exposure``, ``mtm``, ``reported_mtm`` — are pure, row-local ``pl.Expr`` terms in
an internal :class:`~schenberg.core.graph.FormulaGraph`, so they get
``explain`` / ``info`` / ``to_mermaid`` / ``stage`` for free and reuse the proven
formula engine. The only genuinely new machinery is the small, inspectable join
plan; nothing here calls ``.collect()``.

    position_value = (
        PositionView("position_value", output=PositionValue)
        .spine(Position)
        .source("value", InstrumentValue, on=("instrument_type", "instrument_id"))
        .source("book", BookContract, on="book")
        .source("fx", ReportingFx, on=("currency", "reporting_currency"))
    )

    P, V, FX = position_value.position, position_value.value, position_value.fx

    @position_value.measure(symbol="E")
    def exposure(side=uses(P.side), qty=uses(P.quantity)) -> pl.Expr:
        return side * qty

    @position_value.measure(symbol="MTM")
    def mtm(e=uses(exposure), val=uses(V.value)) -> pl.Expr:
        return e * val

    position_value.returns()
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from inspect import Parameter, Signature
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from schenberg.core.fold import Fold

import polars as pl

from schenberg.core.columns import ColumnLike, col_name
from schenberg.core.graph import FormulaGraph, Term, TermKind, Uses


def _keys(on: ColumnLike | Sequence[ColumnLike]) -> tuple[str, ...]:
    if isinstance(on, str) or not isinstance(on, Sequence):
        return (col_name(cast(ColumnLike, on)),)
    return tuple(col_name(cast(ColumnLike, k)) for k in on)


def _schema_columns(schema: Any) -> set[str]:
    return set(schema.to_schema().columns.keys())


def _as_lazy(frame: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    return frame.lazy() if isinstance(frame, pl.DataFrame) else frame


@dataclass(frozen=True, slots=True)
class _Source:
    name: str
    schema: Any
    keys: tuple[str, ...]
    prefix: str | None


class _SourceNamespace:
    """Attribute access over a source schema, yielding graph INPUT ``Term``\\ s.

    ``position_value.value.value`` resolves to the INPUT term for the joined
    ``value`` column, validated against ``InstrumentValue`` at authoring time so a
    typo fails immediately, not at ``collect()``. Non-key columns honour a source
    ``prefix`` (their physical, post-join name)."""

    __slots__ = ("_schema", "_names", "_keys", "_prefix")

    def __init__(
        self, schema: Any, *, keys: tuple[str, ...] = (), prefix: str | None = None
    ) -> None:
        self._schema = schema
        self._names = _schema_columns(schema)
        self._keys = set(keys)
        self._prefix = prefix

    def physical(self, name: str) -> str:
        if self._prefix and name not in self._keys:
            return f"{self._prefix}{name}"
        return name

    def __getattr__(self, name: str) -> Term[Any]:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._names:
            raise AttributeError(f"{name!r} is not a declared column in {self._schema.__name__}")
        return Term(name=self.physical(name), kind=TermKind.INPUT)


@dataclass(frozen=True, slots=True)
class Measure:
    """A reusable measure: a small registrar that adds one term to a view.

    Built by the helpers in :mod:`schenberg.position.measures` (``exposure()``,
    ``mtm()``, ...) — the position-layer analogue of ``sum_``/``first_`` for a
    :class:`~schenberg.core.fold.Fold`. ``view.add(*measures)`` applies them."""

    register: Callable[[PositionView], Term[Any]]


class PositionView:
    """A declarative, lazy view of ``Position × InstrumentValue × Context``.

    Declare the spine and context sources (the join plan), then declare measures
    the same way pricing formulas are declared — with ``@view.measure`` and
    ``uses(...)`` over the source namespaces, or with reusable
    :class:`Measure`\\ s via :meth:`add`. :meth:`compute` interprets the whole
    thing as one lazy Polars query; :meth:`stage` exposes every join column and
    intermediate measure for null-propagation debugging; :meth:`explain`,
    :meth:`info` and :meth:`to_mermaid` describe it. Nothing collects.
    """

    def __init__(self, name: str, *, output: Any | None = None) -> None:
        self.name = name
        self._output = output
        self._spine_name: str | None = None
        self._spine_schema: Any | None = None
        self._sources: list[_Source] = []
        self._ns: dict[str, _SourceNamespace] = {}
        self._measure_names: list[str] = []
        self._g = FormulaGraph(name, input=None)

    # ---- declaration: the join plan --------------------------------------

    def spine(self, schema: Any, *, name: str = "position") -> PositionView:
        """Set the spine: one row per position, carrying the keys. Its namespace
        is exposed as ``view.<name>`` (default ``view.position``)."""
        self._spine_name = name
        self._spine_schema = schema
        self._ns[name] = _SourceNamespace(schema)
        return self

    def source(
        self,
        name: str,
        schema: Any,
        *,
        on: ColumnLike | Sequence[ColumnLike],
        prefix: str | None = None,
    ) -> PositionView:
        """Join a context source by key(s). Sources join in declaration order, so
        a later source may key on a column an earlier one brought in (``fx`` keys
        on the ``reporting_currency`` that ``book`` supplies)."""
        keys = _keys(on)
        self._check_conflicts(name, schema, keys, prefix)
        self._sources.append(_Source(name, schema, keys, prefix))
        self._ns[name] = _SourceNamespace(schema, keys=keys, prefix=prefix)
        return self

    def _check_conflicts(
        self, name: str, schema: Any, keys: tuple[str, ...], prefix: str | None
    ) -> None:
        existing: set[str] = set()
        if self._spine_schema is not None:
            existing |= _schema_columns(self._spine_schema)
        for src in self._sources:
            existing |= {self._ns[src.name].physical(c) for c in _schema_columns(src.schema)}
        new_ns = _SourceNamespace(schema, keys=keys, prefix=prefix)
        non_key = {new_ns.physical(c) for c in _schema_columns(schema) if c not in keys}
        clash = sorted(non_key & existing)
        if clash:
            raise ValueError(
                f"position view {self.name!r}: source {name!r} non-key column(s) {clash} "
                f"collide with already-joined columns; pass a prefix=... or make them join keys"
            )

    # ---- source namespaces (view.position, view.value, ...) --------------

    def __getattr__(self, name: str) -> _SourceNamespace:
        ns = self.__dict__.get("_ns")
        if ns is not None and name in ns:
            return ns[name]
        raise AttributeError(name)

    def ns(self, name: str) -> _SourceNamespace:
        """The namespace for a source (handy when a source name shadows a
        method)."""
        return self._ns[name]

    # ---- declaration: measures -------------------------------------------

    def measure(self, fn: Callable[..., pl.Expr] | None = None, /, **kwargs: Any) -> Any:
        """Register one measure. Works bare (``@view.measure``) or parameterized
        (``@view.measure(name=PV.mtm, symbol=...)``); dependencies come from
        ``uses(...)`` defaults over the source namespaces and earlier measures —
        exactly like a pricing formula. ``name`` may be a typed column reference
        (``cols(PositionValue).mtm``). Returns the measure :class:`Term`."""
        if "name" in kwargs and kwargs["name"] is not None:
            kwargs["name"] = col_name(kwargs["name"])
        if fn is None:
            decorator = self._g.formula(**kwargs)

            def wrapped(real_fn: Callable[..., pl.Expr]) -> Term[Any]:
                term = decorator(real_fn)
                self._measure_names.append(term.name)
                return term

            return wrapped
        term = self._g.formula()(fn)
        self._measure_names.append(term.name)
        return term

    def derive(
        self,
        name: ColumnLike,
        terms: Sequence[Term[Any]],
        fn: Callable[..., pl.Expr],
        **meta: Any,
    ) -> Term[Any]:
        """Register a measure from an explicit list of dependency terms and a
        reducer ``fn(*exprs)`` — for dynamic-arity measures (e.g. a total that
        sums an arbitrary set of components)."""
        measure_name = col_name(name)
        params = [
            Parameter(f"_a{i}", Parameter.POSITIONAL_OR_KEYWORD, default=Uses(term))
            for i, term in enumerate(terms)
        ]

        def wrapper(*args: pl.Expr) -> pl.Expr:
            return fn(*args)

        wrapper.__signature__ = Signature(params)  # ty: ignore[unresolved-attribute]
        wrapper.__name__ = measure_name
        term = self._g.formula(name=measure_name, **meta)(wrapper)
        self._measure_names.append(term.name)
        return term

    def add(self, *measures: Measure) -> PositionView:
        """Apply reusable :class:`Measure`\\ s (from
        :mod:`schenberg.position.measures`)."""
        for measure in measures:
            measure.register(self)
        return self

    def col(self, ref: ColumnLike) -> Term[Any]:
        """Resolve a typed column/measure reference to its :class:`Term` — a
        registered measure, or a joined source column. Accepts a schema column
        (``cols(InstrumentValue).value``), a graph ``Term``, or a plain name."""
        name = col_name(ref)
        if name in self._g._indices:  # an existing measure (or boundary) term
            return self._g._port_term(name)
        for namespace in self._ns.values():
            if name in namespace._names:
                return getattr(namespace, name)
        raise KeyError(f"position view {self.name!r}: {name!r} is not a measure or a source column")

    def by(self, *keys: ColumnLike) -> "Fold":
        """Create a :class:`~schenberg.core.fold.Fold` that groups this view's output
        by *keys* and sums every numeric measure automatically.

        A concise alternative to writing a :class:`~schenberg.core.fold.Fold` by
        hand::

            rollup = position_value.by(PV.book)
            # equivalent to:
            # Fold("...", input_schema=PositionValue).by(PV.book).returns(
            #     None, exposure=sum_(PV.exposure), mtm=sum_(PV.mtm), ...
            # )

        The returned :class:`~schenberg.core.fold.Fold` is lazy and fully
        inspectable via ``.explain()`` / ``.info()``.
        """
        import typing

        from schenberg.core.fold import Fold, sum_

        if self._output is None:
            raise ValueError(
                f"position view {self.name!r} has no output schema; call .returns() first"
            )

        key_names = {col_name(k) for k in keys}
        fields = _schema_columns(self._output)

        unknown = key_names - fields
        if unknown:
            raise ValueError(
                f"position view {self.name!r}: by() key(s) {sorted(unknown)} "
                f"are not columns of {self._output.__name__}"
            )

        hints = typing.get_type_hints(self._output)
        aggs = {
            name: sum_(name)
            for name in fields
            if name not in key_names and hints.get(name) in (float, int)
        }

        key_label = "_".join(col_name(k) for k in keys)
        fold_name = f"{self.name}_by_{key_label}"

        return Fold(fold_name, input_schema=self._output).by(*keys).returns(None, **aggs)

    def returns(self, schema: Any | None = None) -> PositionView:
        """Publish the typed output view. Each field is satisfied *by name* — a
        measure term, or a carried spine/source column — so there is no column
        mapping."""
        resolved = schema if schema is not None else self._output
        if resolved is None:
            raise ValueError(f"position view {self.name!r} has no output schema")
        self._output = resolved
        for field in _schema_columns(resolved):
            if field not in self._g._indices:
                self._g._input_term(field)  # carried-through join column
        self._g.returns("output", resolved)
        return self

    # ---- interpretation (lazy) -------------------------------------------

    def _join(self, spine: pl.LazyFrame | pl.DataFrame, sources: dict[str, Any]) -> pl.LazyFrame:
        lf = _as_lazy(spine)
        for src in self._sources:
            if src.name not in sources:
                raise ValueError(
                    f"position view {self.name!r}: missing source {src.name!r}; "
                    f"pass {src.name}=<frame> to compute(...)"
                )
            right = _as_lazy(sources[src.name])
            if src.prefix:
                right = _prefix_non_key(right, src.prefix, src.keys)
            lf = lf.join(right, on=list(src.keys), how="left")
        return lf

    def compute(
        self,
        spine: pl.LazyFrame | pl.DataFrame,
        *,
        view: str = "output",
        validate: bool = True,
        **sources: pl.LazyFrame | pl.DataFrame,
    ) -> pl.LazyFrame:
        """Interpret the view as one lazy Polars query: join the sources, compile
        every measure, project to the output schema. Validates the output against
        its schema at this (public) boundary unless ``validate=False``. Stays
        lazy."""
        joined = self._join(spine, sources)
        out = self._g.compute(joined, view=view)
        schema = self._g.view_schema(view)
        if schema is not None:
            out = out.select(list(cast(Any, schema).to_schema().columns.keys()))
            if validate:
                out = cast("pl.LazyFrame", cast(Any, schema).validate(out, lazy=True))
        return out

    __call__ = compute

    def stage(
        self, spine: pl.LazyFrame | pl.DataFrame, **sources: pl.LazyFrame | pl.DataFrame
    ) -> pl.LazyFrame:
        """Debug interpretation: the joined frame (every source column) plus every
        measure materialized as its own column. Nulls propagate, so a missing
        instrument value shows up first as a null ``value`` column, with the
        downstream measures null after it. Stays lazy."""
        joined = self._join(spine, sources)
        return self._g.stage(joined, view="output")

    # ---- introspection ---------------------------------------------------

    def measures(self) -> list[str]:
        return list(self._measure_names)

    def join_plan(self) -> list[dict[str, object]]:
        return [
            {"source": src.name, "schema": src.schema.__name__, "on": list(src.keys), "how": "left"}
            for src in self._sources
        ]

    def info(self) -> dict[str, object]:
        return {
            "name": self.name,
            "spine": getattr(self._spine_schema, "__name__", self._spine_schema),
            "sources": self.join_plan(),
            "measures": self.measures(),
            "output": getattr(self._output, "__name__", self._output),
        }

    def explain(self) -> str:
        lines = [f"PositionView {self.name}"]
        oname = getattr(self._output, "__name__", None)
        if oname:
            lines[0] += f" -> {oname}"
        spine_name = getattr(self._spine_schema, "__name__", self._spine_schema)
        lines += ["", "Spine:", f"  - {spine_name}"]
        lines += ["", "Sources (join plan):"]
        for src in self._sources:
            keys = ", ".join(src.keys)
            lines.append(f"  - {src.name} <- {src.schema.__name__}  left on ({keys})")
        lines += ["", "Measures:"]
        for name in self._measure_names:
            lines.append(f"  - {self._g.formula_of(name)}")
        if self._output is not None:
            lines += ["", "Returns:"]
            for col in _schema_columns(self._output):
                lines.append(f"  - {col}")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        lines = ["flowchart LR"]
        spine = getattr(self._spine_schema, "__name__", "spine")
        prev = "spine"
        lines.append(f'    spine[/"{spine} (spine)"/]')
        for i, src in enumerate(self._sources):
            join_id = f"j{i}"
            keys = ", ".join(src.keys)
            lines.append(f'    {src.name}[("{src.schema.__name__}")]')
            lines.append(f'    {prev} --> {join_id}["join ({keys})"]')
            lines.append(f"    {src.name} --> {join_id}")
            prev = join_id
        for name in self._measure_names:
            safe = self._g.formula_of(name).replace('"', "'")
            lines.append(f'    {prev} --> {name}["{safe}"]')
            prev = name
        out = getattr(self._output, "__name__", "output")
        lines.append(f'    {prev} --> output["{out}"]')
        return "\n".join(lines)


def _prefix_non_key(lf: pl.LazyFrame, prefix: str, keys: tuple[str, ...]) -> pl.LazyFrame:
    names = lf.collect_schema().names()
    rename = {n: f"{prefix}{n}" for n in names if n not in keys}
    return lf.rename(rename) if rename else lf
