"""Formula-graph engine: terms, dependencies, and interpretations.

A Schenberg formula is a statically inspectable pricing program. Its values —
inputs, market reads, literals, and derived formulas — are all :class:`Term`\\ s,
wired into an open, typed :class:`FormulaGraph`. The same graph declaration is
*interpreted* many ways: compiled to lazy Polars expressions (:meth:`compute`,
:meth:`stage`), to market requirements, to Mermaid diagrams (:meth:`to_mermaid`),
to explanation text (:meth:`explain`), or to dependency / contract reports
(:meth:`info`). rustworkx owns the topology; Polars owns execution. Nothing here
calls ``.collect()``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, cast

import polars as pl
import rustworkx as rx

from schenberg.core.columns import ColumnRef
from schenberg.core.market import AttachableMarket, MarketDependency, MarketRead, finalize_market

if TYPE_CHECKING:
    from schenberg.market_data.snapshot import MarketSnapshot

FormulaFn = Callable[..., pl.Expr]


class TermKind(StrEnum):
    INPUT = "input"  # a boundary column the caller supplies (fn is None)
    MARKET = "market"  # a boundary column an attached market read supplies
    FORMULA = "formula"  # derived from other terms via fn
    LITERAL = "literal"  # a constant value
    ALIAS = "alias"  # a renamed reference to another term


@dataclass(frozen=True, slots=True)
class Term[T]:
    """One value in a :class:`FormulaGraph`.

    A term is both a graph *node* and a reusable *reference*: ``g.formula`` hands
    one back, ``g.input.spot`` / ``g.market(...)`` expose them, and they flow into
    ``uses(...)`` defaults and ``returns(...)`` mappings. The ``name`` is the
    column the term materializes as; ``kind`` classifies the boundary; ``symbol``
    / ``latex`` / ``description`` drive introspection only and never affect
    execution.
    """

    name: str
    kind: TermKind
    deps: tuple[str, ...] = ()
    fn: FormulaFn | None = None
    dtype: Any | None = None
    value: Any | None = None
    symbol: str | None = None
    latex: str | None = None
    description: str | None = None
    tags: tuple[str, ...] = ()

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class Uses[T]:
    """Dependency marker for a formula parameter. ``uses(term)`` records the graph
    edge while leaving the parameter's ``pl.Expr`` type hint clean for authors."""

    term: Term[T]


def uses[T](term: Term[T]) -> Any:
    """Mark a formula parameter as depending on ``term``.

    Canonical formula style — the default carries the graph dependency, the type
    hint describes what the body sees::

        @g.formula(symbol="T")
        def year_fraction(d: pl.Expr = uses(t.payment_days)) -> pl.Expr:
            return d / 252.0
    """
    return Uses(term)


def term_name(value: object) -> str:
    """Resolve a dependency/mapping value to a column name: ``Uses``, ``Term``,
    ``ColumnRef`` or plain string."""
    if isinstance(value, Uses):
        return value.term.name
    if isinstance(value, Term):
        return value.name
    if isinstance(value, ColumnRef):
        return value.name
    if isinstance(value, str):
        return value
    raise TypeError(f"cannot use {value!r} as a term reference")


@dataclass(frozen=True, slots=True)
class ViewSpec:
    """A named result *view*: the schema it satisfies and the column -> term map."""

    name: str
    schema: object | None
    mapping: dict[str, str]


@dataclass(frozen=True, slots=True)
class GraphInfo:
    name: str
    required_inputs: tuple[str, ...]
    market_inputs: tuple[str, ...]
    market_outputs: tuple[str, ...]
    formula_nodes: tuple[str, ...]
    intermediate_nodes: tuple[str, ...]
    view_nodes: dict[str, str]
    view_dtypes: dict[str, Any | None]


class _Namespace:
    """Attribute access over a graph's boundary terms of one kind."""

    __slots__ = ("_graph", "_names", "_resolve")

    def __init__(
        self,
        graph: FormulaGraph,
        names: set[str] | None,
        resolve: Callable[[str], Term[Any]],
    ) -> None:
        object.__setattr__(self, "_graph", graph)
        object.__setattr__(self, "_names", names)
        object.__setattr__(self, "_resolve", resolve)

    def __getattr__(self, name: str) -> Term[Any]:
        if name.startswith("_"):
            raise AttributeError(name)
        names = object.__getattribute__(self, "_names")
        if names is not None and name not in names:
            raise AttributeError(name)
        return object.__getattribute__(self, "_resolve")(name)


class FormulaGraph:
    """An open, typed, applicative pricing graph.

    Declare a graph over an input schema, name the market data it reads, and wire
    its formulas with explicit term dependencies::

        g = FormulaGraph("generalized_call", input=OptionTrade)
        t = g.input
        m = g.market(
            rate=CurveSpec("curves").value("zero_rate", indexer=t.id_indexador,
                                           tenor=t.payment_days),
            vol=VolSurfaceSpec("vol_surface").implied_vol(indexer=t.id_indexador,
                                           tenor=t.payment_days, strike=t.strike),
        )

        @g.formula(symbol="T", latex=r"\\frac{d}{252}")
        def year_fraction(d: pl.Expr = uses(t.payment_days)) -> pl.Expr:
            return d / 252.0

        g.returns("price", OptionPrice, option_id=t.option_id, price=call_price)

    ``compute(frame, market=..., view="price")`` then compiles one view into a
    single lazy ``with_columns``. The same declaration powers :meth:`explain`,
    :meth:`info`, :meth:`to_mermaid` and :meth:`stage`.
    """

    def __init__(
        self,
        name: str,
        *,
        input: type[Any] | None = None,
    ) -> None:
        self.name = name
        self._graph: rx.PyDiGraph = rx.PyDiGraph(multigraph=False)
        self._indices: dict[str, int] = {}  # term name -> rustworkx index
        self._input_aliases: dict[str, str] = {}  # dep name -> concrete term name
        self._views: dict[str, dict[str, str]] = {}  # view -> {out_col: term_name}
        self._view_schemas: dict[str, object | None] = {}
        self._market: list[MarketDependency] = []
        self._input_schema = input
        self._input_names: set[str] | None = (
            set(cast(Any, input).to_schema().columns.keys()) if input is not None else None
        )

    # ---- boundary namespaces --------------------------------------------

    @property
    def input(self) -> _Namespace:
        """Schema-derived input terms: ``g.input.spot`` returns a stable INPUT
        term. Accessing an undeclared column raises ``AttributeError``."""
        if self._input_schema is None:
            raise AttributeError(
                f"graph {self.name!r} has no input schema; build it with "
                f"FormulaGraph(name, input=Schema)"
            )
        return _Namespace(self, self._input_names, self._input_term)

    @property
    def output(self) -> _Namespace:
        """Output ports: any registered term, addressed by name. Used to bind a
        downstream graph's inputs in :meth:`then`."""
        return _Namespace(self, None, self._port_term)

    def market(self, **reads: AttachableMarket) -> _Namespace:
        """Declare market data as graph terms, naming each output by its keyword.

        Each keyword becomes one MARKET term whose name *is* the output column the
        read writes; the returned namespace exposes them (``m.rate``, ``m.vol``)
        for use in :func:`uses` defaults::

            m = g.market(
                rate=CurveSpec("curves").value("zero_rate", indexer=t.id_indexador,
                                               tenor=t.payment_days),
                vol=VolSurfaceSpec("vol_surface").implied_vol(...),
            )

        A pre-built dependency that writes several columns at once (a multi-output
        join) keeps its own output names; the keyword is then just a label and the
        namespace exposes one term per output column.
        """
        produced: set[str] = set()
        for keyword, read in reads.items():
            if isinstance(read, MarketRead) or len(read.outputs) == 1:
                dependency = finalize_market(read, keyword)
            else:
                dependency = read  # multi-output join keeps its own output names
            self._market.append(dependency)
            for output in dependency.outputs.values():
                self._register_boundary(output, TermKind.MARKET)
                produced.add(output)
        return _Namespace(self, produced, self._port_term)

    def _input_term(self, name: str) -> Term[Any]:
        return self._register_boundary(name, TermKind.INPUT)

    def _port_term(self, name: str) -> Term[Any]:
        idx = self._indices.get(name)
        if idx is None:
            raise AttributeError(f"{name!r} is not a term in graph {self.name!r}")
        return cast(Term[Any], self._graph[idx])

    def _register_boundary(self, name: str, kind: TermKind) -> Term[Any]:
        idx = self._indices.get(name)
        if idx is None:
            term = Term(name=name, kind=kind)
            self._indices[name] = self._graph.add_node(term)
            return term
        existing = cast(Term[Any], self._graph[idx])
        return existing

    # ---- formula construction -------------------------------------------

    def formula(
        self,
        *,
        dtype: Any = pl.Float64,
        tags: tuple[str, ...] = (),
        description: str | None = None,
        name: str | None = None,
        symbol: str | None = None,
        latex: str | None = None,
    ) -> Callable[[FormulaFn], Term[Any]]:
        """Decorator registering a FORMULA term and returning it.

        Dependencies come from the parameter *defaults*: ``uses(term)`` or a bare
        ``Term`` default. The returned term can itself feed later formulas::

            @g.formula(symbol="d_2")
            def d2(d1_: pl.Expr = uses(d1), sigma: pl.Expr = uses(m.vol)) -> pl.Expr:
                return d1_ - sigma

        ``latex`` is a pure-introspection label; it never affects execution.
        """

        def register(fn: FormulaFn) -> Term[Any]:
            term_id = name or fn.__name__  # ty: ignore[unresolved-attribute]
            term = Term(
                name=term_id,
                kind=TermKind.FORMULA,
                deps=self._deps_from_signature(fn, term_id),
                fn=fn,
                dtype=dtype,
                tags=tuple(tags),
                description=description,
                symbol=symbol,
                latex=latex,
            )
            self._register(term)
            return term

        return register

    def _deps_from_signature(self, fn: FormulaFn, term_id: str) -> tuple[str, ...]:
        deps: list[str] = []
        for pname, param in signature(fn).parameters.items():
            default = param.default
            if isinstance(default, Uses):
                deps.append(default.term.name)
            elif isinstance(default, Term):
                deps.append(default.name)
            elif default is Parameter.empty:
                raise ValueError(
                    f"formula {term_id} parameter {pname!r} has no Term dependency.\n"
                    f"Use {pname}: pl.Expr = uses(<term>), or pass a Term default."
                )
            else:
                raise ValueError(
                    f"formula {term_id} parameter {pname!r} has an unexpected default "
                    f"{default!r}; use uses(<term>) or a bare Term"
                )
        return tuple(deps)

    def _register(self, term: Term[Any]) -> None:
        if term.name in self._indices and self._graph[self._indices[term.name]].fn is not None:
            raise ValueError(f"term {term.name!r} already defined in graph {self.name!r}")
        if term.name in self._indices:  # a boundary term promoted to a formula
            self._graph[self._indices[term.name]] = term
        else:
            self._indices[term.name] = self._graph.add_node(term)
        idx = self._indices[term.name]
        for dep in term.deps:
            self._graph.add_edge(self._ensure_input(dep), idx, None)  # dep -> dependent
        self._validate_dag()

    def _ensure_input(self, name: str) -> int:
        idx = self._indices.get(name)
        if idx is None:
            idx = self._graph.add_node(Term(name, TermKind.INPUT))
            self._indices[name] = idx
        return idx

    def _validate_dag(self) -> None:
        if not rx.is_directed_acyclic_graph(self._graph):
            raise ValueError(f"cycle detected in graph {self.name!r}")

    def returns(
        self,
        view: str,
        schema: object | None = None,
        /,
        **mapping: object,
    ) -> FormulaGraph:
        """Declare a result *view*: ``out_column -> term``.

        Values may be terms (canonical), ``uses(...)`` markers, ``ColumnRef``\\ s
        or plain strings::

            g.returns("price", OptionPrice, option_id=t.option_id, price=call_price)

        With a ``schema`` (any object exposing ``.to_schema().columns.keys()``)
        every field must be satisfied: pass it explicitly, or let an identically
        named term fill it. Extra columns not in the schema are rejected.
        """
        resolved = {col: term_name(value) for col, value in mapping.items()}
        if schema is not None:
            fields = list(cast(Any, schema).to_schema().columns.keys())
            extra = sorted(set(resolved) - set(fields))
            if extra:
                raise ValueError(f"view {view!r} maps columns not in schema: {extra}")
            full = {f: resolved.get(f, f) for f in fields}
        else:
            full = resolved
        self._views[view] = full
        self._view_schemas[view] = schema
        return self

    # ---- composition (open-graph semantics) -----------------------------

    @classmethod
    def identity(cls, name: str = "identity") -> FormulaGraph:
        """The empty graph: a unit for :meth:`then` and :meth:`merge`."""
        return cls(name)

    @classmethod
    def compose(cls, name: str, *graphs: FormulaGraph) -> FormulaGraph:  # noqa: PLR0912
        """Parallel composition: merge graphs in the same environment.

        Formula terms are shared by name when identical and conflict when a name
        names two different formulas. Boundary terms (input/market) are recreated,
        so an INPUT in one graph satisfied by a FORMULA in another resolves. Views,
        market reads and aliases carry through; a re-declared view must not
        conflict.
        """
        merged = cls(name)
        formulas: dict[str, Term[Any]] = {}
        boundary_kinds: dict[str, TermKind] = {}
        for g in graphs:
            for idx in g._indices.values():
                node = cast(Term[Any], g._graph[idx])
                if node.fn is None:
                    boundary_kinds.setdefault(node.name, node.kind)
                    continue
                prev = formulas.get(node.name)
                if prev is not None and prev != node:
                    raise ValueError(f"conflicting formula {node.name!r} in compose({name!r})")
                formulas[node.name] = node
            merged._input_aliases.update(g._input_aliases)
            merged._market.extend(g._market)
            for view, mapping in g._views.items():
                existing = merged._views.setdefault(view, {})
                for col, tname in mapping.items():
                    if existing.get(col, tname) != tname:
                        raise ValueError(
                            f"conflicting view column {view}.{col} in compose({name!r})"
                        )
                    existing[col] = tname
                merged._view_schemas.setdefault(view, g._view_schemas.get(view))

        for term in formulas.values():  # 1. formula terms
            merged._indices[term.name] = merged._graph.add_node(term)
        for name_, kind in boundary_kinds.items():  # 2. boundary terms for deps
            if name_ not in formulas:
                merged._register_boundary(name_, kind)
        for term in formulas.values():  # 3. inputs for any remaining deps
            for dep in term.deps:
                merged._ensure_input(dep)
        for term in formulas.values():  # 4. edges dep -> dependent
            for dep in term.deps:
                merged._graph.add_edge(merged._indices[dep], merged._indices[term.name], None)
        merged._validate_dag()
        return merged

    def merge(self, *others: FormulaGraph, name: str | None = None) -> FormulaGraph:
        """Parallel composition of ``self`` with ``others`` (same environment)."""
        return type(self).compose(name or self.name, self, *others)

    def extend(self, *others: FormulaGraph, name: str | None = None) -> FormulaGraph:
        """Same-environment formula extension — add formula blocks that read the
        same inputs/market. The common case for layering Greeks onto a price."""
        return type(self).compose(name or self.name, self, *others)

    def compose_with(self, *others: FormulaGraph, name: str | None = None) -> FormulaGraph:
        """Friendly alias for :meth:`extend` / :meth:`merge`."""
        return type(self).compose(name or self.name, self, *others)

    def then(
        self,
        other: FormulaGraph,
        *,
        bind: Mapping[Term[Any], Term[Any]] | None = None,
        name: str | None = None,
    ) -> FormulaGraph:
        """Sequential composition by ports: feed ``self``'s outputs into ``other``'s
        inputs. ``bind`` maps each downstream input term to the upstream term that
        supplies it::

            priced = payoff.then(discounting,
                                 bind={discounting.input.future_value: payoff.output.future_value})
        """
        merged = type(self).compose(name or self.name, self, other)
        for target, source in (bind or {}).items():
            merged._input_aliases[term_name(target)] = term_name(source)
        merged._validate_dag()
        return merged

    # ---- compilation -----------------------------------------------------

    def _resolve(self, name: str) -> str:
        seen: set[str] = set()
        while (
            name in self._input_aliases and self._input_aliases[name] != name and name not in seen
        ):
            seen.add(name)
            name = self._input_aliases[name]
        return name

    def expr(
        self,
        target: str,
        provided: Mapping[str, pl.Expr] | None = None,
        _cache: dict[str, pl.Expr] | None = None,
    ) -> pl.Expr:
        """Recursively compile ``target`` into one nested ``pl.Expr``."""
        cache = _cache if _cache is not None else {}
        if target in cache:
            return cache[target]
        if provided and target in provided:
            return cache.setdefault(target, provided[target])
        alias = self._input_aliases.get(target)
        if alias is not None and alias != target:
            return cache.setdefault(target, self.expr(alias, provided, cache))
        idx = self._indices.get(target)
        if idx is None:
            raise KeyError(
                f"{target!r} is not a term in graph {self.name!r}; "
                f"known terms: {sorted(self._indices)}"
            )
        term = cast(Term[Any], self._graph[idx])
        if term.fn is None:  # boundary column
            result = pl.col(target)
        else:
            args = [self.expr(dep, provided, cache) for dep in term.deps]
            result = term.fn(*args)
        cache[target] = result
        return result

    def compute(
        self,
        frame: pl.LazyFrame,
        *,
        market: MarketSnapshot | None = None,
        outputs: Mapping[str, str] | None = None,
        view: str | None = None,
    ) -> pl.LazyFrame:
        """Interpret the graph as lazy Polars: attach market data, compile each
        view column to a nested expression, add them with one ``with_columns``.
        Stays lazy."""
        frame = self._attach_market(frame, market)
        mapping = self._resolve_view(outputs, view)
        frame = self._with_missing_inputs(frame, mapping.values())
        cache: dict[str, pl.Expr] = {}
        columns = [self.expr(node, _cache=cache).alias(col) for col, node in mapping.items()]
        return frame.with_columns(columns)

    def stage(
        self,
        frame: pl.LazyFrame,
        *,
        market: MarketSnapshot | None = None,
        view: str | None = None,
        targets: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Staged-debug interpretation: materialize every intermediate term as its
        own column. Wider/slower than :meth:`compute`, but every step is
        inspectable (nulls propagate, so the first unexpectedly-null column is the
        root cause). Stays lazy."""
        frame = self._attach_market(frame, market)
        if targets is None:
            if view is None:
                raise ValueError("pass targets=[...] or view=...")
            targets = list(self._views[view].values())

        order: list[str] = []
        seen: set[str] = set()

        def emit(name: str) -> None:
            real = self._resolve(name)
            if real in seen:
                return
            seen.add(real)
            term = cast(Term[Any], self._graph[self._indices[real]])
            if term.fn is None:
                return
            for dep in term.deps:
                emit(dep)
            order.append(real)

        for t in targets:
            emit(t)

        for name in order:
            term = cast(Term[Any], self._graph[self._indices[name]])
            cols = [pl.col(self._resolve(d)) for d in term.deps]
            assert term.fn is not None
            frame = frame.with_columns(term.fn(*cols).alias(name))
        return frame

    def _attach_market(self, lf: pl.LazyFrame, market: MarketSnapshot | None) -> pl.LazyFrame:
        if self._market:
            if market is None:
                raise ValueError(f"graph {self.name!r} needs a MarketSnapshot")
            for req in self._market:
                lf = market.attach(lf, req)
        return lf

    def _with_missing_inputs(self, lf: pl.LazyFrame, targets: Iterable[str]) -> pl.LazyFrame:
        required = {
            dep
            for target in targets
            for dep in self.dependencies_of(target) | {target}
            if self._graph[self._indices[dep]].fn is None
        }
        if not required:
            return lf
        present = set(lf.collect_schema().names())
        missing = sorted(required - present)
        if not missing:
            return lf
        return lf.with_columns(pl.lit(None).alias(name) for name in missing)

    def _resolve_view(self, outputs: Mapping[str, str] | None, view: str | None) -> dict[str, str]:
        if outputs is not None:
            return dict(outputs)
        if view is not None:
            try:
                return dict(self._views[view])
            except KeyError:
                raise KeyError(f"no view {view!r} in graph {self.name!r}") from None
        raise ValueError("pass view=...")

    # ---- shared computation interface -----------------------------------

    def has_view(self, view: str) -> bool:
        return view in self._views

    def view_schema(self, view: str) -> object | None:
        return self._view_schemas.get(view)

    # ---- introspection ---------------------------------------------------

    def edges(self) -> list[tuple[str, str]]:
        names = {i: self._graph[i].name for i in self._indices.values()}
        return [(names[a], names[b]) for a, b in self._graph.edge_list()]

    def dependencies_of(self, target: str | Term[Any]) -> set[str]:
        idx = self._indices[term_name(target)]
        return {self._graph[i].name for i in rx.ancestors(self._graph, idx)}

    def topological_order(self) -> list[str]:
        return [self._graph[i].name for i in rx.topological_sort(self._graph)]

    def describe(self, target: str | Term[Any]) -> str:
        term = cast(Term[Any], self._graph[self._indices[term_name(target)]])
        return (
            f"{term.name} [{term.kind}] tags={term.tags or '()'}\n"
            f"  deps: {', '.join(term.deps) or '-'}\n"
            f"  {term.description or ''}"
        ).rstrip()

    def formula_of(self, target: str | Term[Any]) -> str:
        term = cast(Term[Any], self._graph[self._indices[term_name(target)]])
        if term.fn is None:
            return term.symbol or term.name
        lhs = term.symbol or term.name
        if term.latex:
            return f"{lhs} = {term.latex}"
        deps = ", ".join(term.deps)
        return f"{lhs} = \\operatorname{{{term.name}}}({deps})"

    def formulas(self) -> dict[str, str]:
        return {
            name: self.formula_of(name)
            for name, idx in self._indices.items()
            if self._graph[idx].kind is TermKind.FORMULA
        }

    def _market_outputs(self) -> set[str]:
        return {out for req in self._market for out in req.outputs.values()}

    def _kind_of(self, name: str, market_outputs: set[str]) -> str:
        if name in market_outputs:
            return "market"
        kind = self._graph[self._indices[name]].kind
        if kind is TermKind.FORMULA:
            return "formula"
        if kind is TermKind.MARKET:
            return "market"
        return "input"

    def to_mermaid(
        self,
        *,
        math_labels: bool = False,
        show_kinds: bool = False,
        view: str | None = None,
    ) -> str:
        """Interpret the graph as a Mermaid flowchart, distinguishing input,
        market, formula and view/output terms."""
        view_nodes = set(self._views.get(view, {}).values()) if view else set()
        market_outputs = self._market_outputs()

        def label(name: str) -> str:
            if not math_labels:
                return name
            return self.formula_of(name).replace('"', "'")

        lines = ["flowchart LR"]
        for a, b in self.edges():
            if math_labels:
                lines.append(f'    {a}["{label(a)}"] --> {b}["{label(b)}"]')
            else:
                lines.append(f"    {a} --> {b}")
        if show_kinds:
            for name in self._indices:
                classes = [self._kind_of(name, market_outputs)]
                if name in view_nodes:
                    classes.append("output")
                for cls in classes:
                    lines.append(f"    class {name} {cls}")
        return "\n".join(lines)

    def info(self, *, view: str | None = None) -> GraphInfo:
        view_nodes = self._resolve_view(None, view) if view else {}
        selected = set(view_nodes.values())
        formula_nodes = tuple(
            name
            for name in self.topological_order()
            if self._graph[self._indices[name]].kind is TermKind.FORMULA
        )
        return GraphInfo(
            name=self.name,
            required_inputs=tuple(sorted(self.required_inputs())),
            market_inputs=tuple(sorted({k for req in self._market for k in req.left_keys})),
            market_outputs=tuple(sorted(self._market_outputs())),
            formula_nodes=formula_nodes,
            intermediate_nodes=tuple(name for name in formula_nodes if name not in selected),
            view_nodes=view_nodes,
            view_dtypes=self.view_dtypes(view) if view else {},
        )

    def _market_report(self) -> list[str]:
        lines: list[str] = []
        for req in self._market:
            keys = ", ".join(req.left_keys)
            for out in req.outputs.values():
                lines.append(f"  - {out} <- {req.table}({keys})")
        return lines

    def explain(self, target: str | None = None, *, view: str | None = None) -> str:
        """Interpret the graph as explanation text: its inputs, market reads,
        formula path and returns."""
        if target is None and view is None:
            raise ValueError("pass target=... or view=...")
        if target is not None:
            targets = [target]
        else:
            assert view is not None
            targets = list(self._views[view].values())
        needed = set(targets)
        for item in targets:
            needed |= self.dependencies_of(item)
        path = [
            name
            for name in self.topological_order()
            if name in needed and self._graph[self._indices[name]].kind is TermKind.FORMULA
        ]
        info = self.info(view=view)
        lines = [f"FormulaGraph {self.name}"]
        if view is not None:
            schema = self._view_schemas.get(view)
            sname = getattr(schema, "__name__", None)
            lines.append("")
            lines.append("View:")
            lines.append(f"  - {view}" + (f" -> {sname}" if sname else ""))
        lines.append("")
        lines.append("Inputs: " + (", ".join(info.required_inputs) or "-"))
        market = self._market_report()
        lines.append("")
        lines.append("Market reads:")
        lines.extend(market or ["  - (none)"])
        lines.append("")
        lines.append("Formulas:")
        for name in path:
            desc = self._graph[self._indices[name]].description
            suffix = f"  # {desc}" if desc else ""
            lines.append(f"  - {self.formula_of(name)}{suffix}")
        if view is not None:
            lines.append("")
            lines.append("Returns:")
            for col, tname in self._views[view].items():
                lines.append(f"  - {col} <- {tname}")
        return "\n".join(lines)

    def validate_view(self, view: str, schema: object) -> None:
        mapping = self._resolve_view(None, view)
        fields = set(cast(Any, schema).to_schema().columns.keys())
        missing = sorted(fields - set(mapping))
        if missing:
            raise ValueError(f"view {view!r} is missing schema fields: {missing}")
        unknown = sorted(node for node in mapping.values() if node not in self._indices)
        if unknown:
            raise ValueError(f"view {view!r} maps to unknown terms: {unknown}")

    def required_inputs(self) -> set[str]:
        """The columns a caller must supply: leaf boundary terms, minus what market
        reads provide and aliases redirect, plus market join keys."""
        graph_inputs = {n for n, i in self._indices.items() if self._graph[i].fn is None}
        market_provided = self._market_outputs()
        market_keys = {k for req in self._market for k in req.left_keys}
        return (graph_inputs - market_provided - set(self._input_aliases)) | market_keys

    def view_dtypes(self, view: str) -> dict[str, Any | None]:
        return {
            col: self._graph[self._indices[node]].dtype for col, node in self._views[view].items()
        }


# ---- contract-oriented facade -------------------------------------------------

_OUTPUT_VIEW = "output"


@dataclass(frozen=True, slots=True)
class Bound:
    """The pair :meth:`PricingGraph.bind` resolves: the trade frame and the market
    environment its formulas will read. Construction stays lazy; nothing collects."""

    frame: pl.LazyFrame
    market: MarketSnapshot | None


class _PricingGraphFactory:
    """What ``PricingGraph[Contract, Requirements, Output]`` evaluates to: a callable
    that remembers the three type arguments and builds the graph from a name."""

    __slots__ = ("_contract", "_requirements", "_output")

    def __init__(self, contract: Any, requirements: Any, output: Any) -> None:
        self._contract = contract
        self._requirements = requirements
        self._output = output

    def __call__(self, name: str) -> PricingGraph:
        return PricingGraph(
            name,
            contract=self._contract,
            requirements=self._requirements,
            output=self._output,
        )


class PricingGraph:
    """A pure, contract-oriented pricing graph over a triple of boundary schemas.

    ``PricingGraph[Contract, Requirements, Output](name)`` builds a graph whose
    inputs are ``Contract`` columns, whose market columns are declared by a
    :class:`~schenberg.market_data.requirements.MarketRequirements` subclass, and
    whose primary result satisfies ``Output``. Formulas read ``g.contract`` and
    ``g.market`` terms and never join; :meth:`bind` resolves the market environment
    and :meth:`plan` returns the lazy Polars plan for the whole instrument.

    A graph publishes its primary ``output`` view with :meth:`returns` and any
    number of secondary typed views with :meth:`view` (e.g. an option's ``price``
    and ``state``). Every view is satisfied *by name*: a schema field is filled by
    the contract, market or formula term of the same name -- no column mapping. As
    a :class:`Computation` (``compute`` / ``has_view`` / ``view_schema``) it slots
    directly into a :class:`~schenberg.core.router.Router` or
    :class:`~schenberg.core.structure.Structure`.

    It is a typed face over :class:`FormulaGraph` -- the private engine that owns
    topology and Polars compilation; the requirements compile to the same
    :class:`MarketDependency` objects the engine attaches.
    """

    def __init__(
        self,
        name: str,
        *,
        contract: type[Any] | None = None,
        requirements: Any | None = None,
        output: type[Any] | None = None,
    ) -> None:
        self.name = name
        self._contract = contract
        self._output = output
        self._g = FormulaGraph(name, input=contract)
        self._contract_names = (
            set(cast(Any, contract).to_schema().columns.keys()) if contract is not None else set()
        )
        reqs: dict[str, MarketDependency] = (
            dict(getattr(requirements, "__requirements__", {})) if requirements is not None else {}
        )
        self._market_ns: _Namespace | None = self._g.market(**reqs) if reqs else None

    def __class_getitem__(cls, item: tuple[Any, Any, Any]) -> _PricingGraphFactory:
        contract, requirements, output = item
        return _PricingGraphFactory(contract, requirements, output)

    @property
    def contract(self) -> _Namespace:
        """Contract columns as INPUT terms: ``g.contract.payment_days``."""
        return self._g.input

    @property
    def market(self) -> _Namespace:
        """Declared market columns as MARKET terms: ``g.market.zero_rate``."""
        if self._market_ns is None:
            raise AttributeError(f"graph {self.name!r} declares no market requirements")
        return self._market_ns

    def formula(self, fn: FormulaFn | None = None, /, **kwargs: Any) -> Any:
        """Register a formula. Works bare (``@g.formula``) or parameterized
        (``@g.formula(symbol=...)``); dependencies come from ``uses(...)`` defaults."""
        if fn is None:
            return self._g.formula(**kwargs)
        return self._g.formula()(fn)

    def returns(self, schema: type[Any] | None = None) -> PricingGraph:
        """Publish the primary ``output`` view, matching its schema's fields to
        like-named terms. Defaults to the ``Output`` type parameter."""
        schema = schema if schema is not None else self._output
        if schema is None:
            raise ValueError(f"graph {self.name!r} has no output schema")
        self._output = schema
        return self.view(_OUTPUT_VIEW, schema)

    def view(self, name: str, schema: type[Any]) -> PricingGraph:
        """Publish a secondary typed view (``g.view("price", OptionPrice)``).

        Like :meth:`returns`, every field is satisfied by the term of the same name
        -- contract column, market column or formula -- so no mapping is written.
        """
        fields = list(cast(Any, schema).to_schema().columns.keys())
        for column in fields:
            if column not in self._g._indices and column in self._contract_names:
                self._g._input_term(column)  # pass-through contract column
        self._g.returns(name, schema)
        return self

    def bind(
        self,
        trades: pl.LazyFrame | pl.DataFrame,
        *,
        market: MarketSnapshot | None = None,
    ) -> Bound:
        """Resolve the market environment for a set of trades. Stays lazy."""
        frame = trades.lazy() if isinstance(trades, pl.DataFrame) else trades
        return Bound(frame=frame, market=market)

    def plan(self, bound: Bound, *, view: str = _OUTPUT_VIEW) -> pl.LazyFrame:
        """The lazy Polars plan for the bound trades, projected to a view's schema."""
        schema = self._g.view_schema(view)
        planned = self._g.compute(bound.frame, market=bound.market, view=view)
        if schema is None:
            return planned
        return planned.select(list(cast(Any, schema).to_schema().columns.keys()))

    # ---- Computation protocol (router / structure branch) ----------------
    def compute(
        self,
        frame: pl.LazyFrame,
        *,
        market: MarketSnapshot | None = None,
        view: str = _OUTPUT_VIEW,
    ) -> pl.LazyFrame:
        """Interpret a view as lazy Polars, carrying input columns through (for a
        router/structure to weight, fold or normalize). Stays lazy."""
        return self._g.compute(frame, market=market, view=view)

    def has_view(self, view: str) -> bool:
        return self._g.has_view(view)

    def view_schema(self, view: str) -> object | None:
        return self._g.view_schema(view)

    # Introspection passthroughs -- the same declaration explains itself.
    def explain(self, *, view: str = _OUTPUT_VIEW, **kwargs: Any) -> str:
        return self._g.explain(view=view, **kwargs)

    def to_mermaid(self, *, view: str = _OUTPUT_VIEW, **kwargs: Any) -> str:
        return self._g.to_mermaid(view=view, **kwargs)
