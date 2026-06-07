"""AST formula graph: a pure, statically inspectable pricing program.

A Schenberg formula is an :class:`~schenberg.core.expr.Expr` tree. Inputs are
``var`` nodes; derived values are named expressions registered with :meth:`let`.
The graph never joins, never reads market data, never sees a snapshot — market
columns arrive pre-resolved as input columns (see :func:`schenberg.market_data.roles.bind`).

The same declaration is interpreted many ways: compiled to a single lazy Polars
``with_columns`` (:meth:`compute`), rendered to LaTeX *derived from the formula*
(:meth:`formula_of`), to a Mermaid diagram, or to dependency/contract reports.
Nothing here calls ``collect``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any, cast

import polars as pl

from schenberg.core.expr import Expr, compile_polars, lit, to_latex, var


def term_name(value: object) -> str:
    """Resolve a reference to a column/term name: an :class:`Expr` ``var``, an
    object exposing ``.name``, or a plain string."""
    if isinstance(value, Expr):
        if value.op == "var" and value.name is not None:
            return value.name
        raise TypeError(f"cannot use non-var Expr {value!r} as a term reference")
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(value, str):
        return value
    raise TypeError(f"cannot use {value!r} as a term reference")


@dataclass(frozen=True, slots=True)
class TermMeta:
    """Introspection-only metadata for a derived term."""

    name: str
    symbol: str | None = None
    description: str | None = None
    tags: tuple[str, ...] = ()
    dtype: Any | None = None


@dataclass(frozen=True, slots=True)
class GraphInfo:
    name: str
    required_inputs: tuple[str, ...]
    formula_nodes: tuple[str, ...]
    intermediate_nodes: tuple[str, ...]
    view_nodes: dict[str, str]
    view_dtypes: dict[str, Any | None]


def _iter_vars(e: Expr) -> Iterator[str]:
    if e.op == "var" and e.name is not None:
        yield e.name
    for arg in e.args:
        yield from _iter_vars(arg)


class _InputNS:
    """Attribute access over a graph's input columns, yielding ``var`` nodes."""

    __slots__ = ("_names",)

    def __init__(self, names: set[str] | None) -> None:
        self._names = names

    def __getattr__(self, name: str) -> Expr:
        if name.startswith("_"):
            raise AttributeError(name)
        if self._names is not None and name not in self._names:
            raise AttributeError(f"{name!r} is not an input column")
        return var(name)


class FormulaGraph:
    """The pure, typed, AST-based pricing engine.

        g = FormulaGraph("forward", input=ForwardPricingInput)

        @g.formula(symbol="T")
        def year_fraction(payment_days):
            return payment_days / 252.0

        @g.formula(symbol="DF")
        def discount_factor(risk_free, year_fraction):
            return exp(-risk_free * year_fraction)

        @g.formula(symbol="FV")
        def future_value(forward_price, strike):
            return forward_price - strike

        @g.formula(symbol="PV")
        def present_value(future_value, discount_factor):
            return future_value * discount_factor

        g.returns("output", ForwardPricing)

    Each parameter is a headless dependency resolved to an input column or a
    prior term. ``plan(frame, view="output")`` compiles each view column to one
    nested expression and adds them with a single ``with_columns``. ``g.let(...)``
    remains the lower-level primitive when an :class:`Expr` is built by hand.
    """

    def __init__(self, name: str, *, input: type[Any] | None = None) -> None:
        self.name = name
        self._input_schema = input
        self._input_names: set[str] | None = (
            set(cast(Any, input).to_schema().columns.keys()) if input is not None else None
        )
        self._terms: dict[str, Expr] = {}
        self._meta: dict[str, TermMeta] = {}
        self._views: dict[str, dict[str, str]] = {}
        self._view_schemas: dict[str, object | None] = {}

    # ---- boundary --------------------------------------------------------

    @property
    def input(self) -> _InputNS:
        return _InputNS(self._input_names)

    contract = input

    # ---- declaration -----------------------------------------------------

    def let(
        self,
        name: str,
        expr: Expr | float | int,
        *,
        symbol: str | None = None,
        description: str | None = None,
        tags: tuple[str, ...] = (),
        dtype: Any = pl.Float64,
    ) -> Expr:
        """Register a derived term ``name = expr`` and return a ``var(name)``
        reference to it. ``expr`` may use input vars and earlier terms."""
        if name in self._terms:
            raise ValueError(f"term {name!r} already defined in graph {self.name!r}")
        if self._input_names is not None and name in self._input_names:
            raise ValueError(f"term {name!r} shadows an input column in graph {self.name!r}")
        node = expr if isinstance(expr, Expr) else lit(expr)
        self._assert_acyclic(name, node)
        self._terms[name] = node
        self._meta[name] = TermMeta(name, symbol, description, tuple(tags), dtype)
        return var(name)

    def formula(
        self,
        *,
        name: str | None = None,
        symbol: str | None = None,
        description: str | None = None,
        tags: Iterable[str] = (),
        dtype: Any = pl.Float64,
    ) -> Callable[[Callable[..., Expr | float | int]], Callable[..., Expr | float | int]]:
        """Decorate a Python function and register its symbolic result as a term.

        Dependencies are declared as **headless parameters**: each argument name
        is resolved to a symbolic ``var`` — from an already-declared term first,
        otherwise from the graph's input schema. So a formula reads like the math
        it represents::

            @g.formula(symbol="PV")
            def present_value(future_value, discount_factor):
                return future_value * discount_factor

        ``future_value`` and ``discount_factor`` are prior terms; a parameter such
        as ``spot`` or ``strike`` resolves to the input column of the same name.
        When an input schema is declared, an unknown parameter fails fast (it is
        neither a prior term nor an input column), catching typos at definition
        time. The legacy namespace names ``c``, ``contract``, ``input`` and
        ``inputs`` still receive the whole input namespace for ``c.<col>`` access,
        but headless parameters are the preferred public style.

        The decorated function must return a Schenberg :class:`Expr` (or a literal
        number), so the graph stays symbolic, inspectable, and lazily compiled.
        """

        def decorator(fn: Callable[..., Expr | float | int]) -> Callable[..., Expr | float | int]:
            fn_name = getattr(fn, "__name__", repr(fn))
            term = name or fn_name
            kwargs: dict[str, object] = {}
            for param_name in inspect.signature(fn).parameters:
                if param_name in self._terms:
                    kwargs[param_name] = var(param_name)
                elif param_name in {"c", "contract", "input", "inputs"}:
                    kwargs[param_name] = self.input
                elif self._input_names is None or param_name in self._input_names:
                    kwargs[param_name] = var(param_name)
                else:
                    raise ValueError(
                        f"unknown formula dependency {param_name!r} in formula {fn_name!r}: "
                        "not a prior term or an input column of "
                        f"{self.name!r}"
                    )
            self.let(
                term,
                fn(**kwargs),
                symbol=symbol,
                description=description,
                tags=tuple(tags),
                dtype=dtype,
            )
            return fn

        return decorator

    def _assert_acyclic(self, name: str, node: Expr) -> None:
        if name in set(_iter_vars(node)):
            raise ValueError(f"term {name!r} references itself")

    def returns(
        self,
        view: str,
        schema: object | None = None,
        /,
        **mapping: object,
    ) -> FormulaGraph:
        """Declare a result view: ``out_column -> term``. With a ``schema`` every
        field is satisfied by an explicit mapping or an identically named term /
        input column; extra columns are rejected."""
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

    # ---- compilation -----------------------------------------------------

    def _inline(self, e: Expr, stack: tuple[str, ...] = ()) -> Expr:
        if e.op == "var" and e.name in self._terms:
            if e.name in stack:
                raise ValueError(f"cycle detected at {e.name!r} in graph {self.name!r}")
            return self._inline(self._terms[e.name], (*stack, e.name))
        if e.args:
            return replace(e, args=tuple(self._inline(a, stack) for a in e.args))
        return e

    def _compile(self, name: str) -> pl.Expr:
        if name in self._terms:
            return compile_polars(self._inline(self._terms[name]))
        return pl.col(name)

    def plan(
        self,
        frame: pl.LazyFrame,
        *,
        outputs: Mapping[str, str] | None = None,
        view: str | None = None,
        materialize_terms: bool = True,
    ) -> pl.LazyFrame:
        """Interpret the graph as a lazy output view.

        By default reachable terms are computed once in topological order, then
        the requested output columns are selected. Passing
        ``materialize_terms=False`` keeps the legacy recursive inlining path. The
        graph never reads market data — market columns arrive pre-resolved as
        input columns. Fails fast when required input columns are absent and stays
        lazy.
        """
        if materialize_terms:
            return self._plan_materialized(frame, outputs=outputs, view=view)
        return self._plan_inlined(frame, outputs=outputs, view=view)

    def _plan_inlined(
        self,
        frame: pl.LazyFrame,
        *,
        outputs: Mapping[str, str] | None = None,
        view: str | None = None,
    ) -> pl.LazyFrame:
        mapping = self._resolve_view(outputs, view)
        self._require_inputs(frame, mapping.values())
        columns = [self._compile(name).alias(col) for col, name in mapping.items()]
        return frame.with_columns(columns)

    def _plan_materialized(
        self,
        frame: pl.LazyFrame,
        *,
        outputs: Mapping[str, str] | None = None,
        view: str | None = None,
    ) -> pl.LazyFrame:
        mapping = self._resolve_view(outputs, view)
        targets = list(mapping.values())
        self._require_inputs(frame, targets)

        reachable = self._reachable(targets)
        order = [name for name in self.topological_order() if name in reachable]
        for name in order:
            frame = frame.with_columns(compile_polars(self._terms[name]).alias(name))

        available = set(frame.collect_schema().names())
        columns = []
        for out_col, term_name in mapping.items():
            if term_name in self._terms or term_name in available:
                columns.append(pl.col(term_name).alias(out_col))
            else:
                columns.append(self._compile(term_name).alias(out_col))
        return frame.select(columns)

    def stage(
        self,
        frame: pl.LazyFrame,
        *,
        view: str | None = None,
        targets: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Debug interpretation: materialize every intermediate term as its own
        column, in dependency order. Wider/slower than :meth:`compute`."""
        if targets is None:
            if view is None:
                raise ValueError("pass targets=[...] or view=...")
            targets = list(self._views[view].values())
        order = [name for name in self.topological_order() if name in self._reachable(targets)]
        self._require_inputs(frame, self.required_inputs_for(targets))
        for name in order:
            frame = frame.with_columns(compile_polars(self._terms[name]).alias(name))
        return frame

    def _resolve_view(self, outputs: Mapping[str, str] | None, view: str | None) -> dict[str, str]:
        if outputs is not None:
            return dict(outputs)
        if view is not None:
            try:
                return dict(self._views[view])
            except KeyError:
                raise KeyError(f"no view {view!r} in graph {self.name!r}") from None
        raise ValueError("pass view=...")

    def _require_inputs(self, lf: pl.LazyFrame, targets: Iterable[str]) -> None:
        needed = self.required_inputs_for(targets)
        present = set(lf.collect_schema().names())
        missing = sorted(needed - present)
        if missing:
            raise ValueError(f"graph {self.name!r} is missing required input column(s): {missing}")

    # ---- dependency analysis ---------------------------------------------

    def _reachable(self, targets: Iterable[str]) -> set[str]:
        out: set[str] = set()
        stack = [t for t in targets if t in self._terms]
        while stack:
            name = stack.pop()
            if name in out:
                continue
            out.add(name)
            for dep in _iter_vars(self._terms[name]):
                if dep in self._terms and dep not in out:
                    stack.append(dep)
        return out

    def required_inputs_for(self, targets: Iterable[str]) -> set[str]:
        needed: set[str] = set()
        for target in targets:
            if target in self._terms:
                needed |= set(_iter_vars(self._inline(self._terms[target])))
            else:
                needed.add(target)
        return needed

    def required_inputs(self, view: str | None = None) -> set[str]:
        targets = list(self._views[view].values()) if view else list(self._views)
        if view is None:
            targets = [t for mapping in self._views.values() for t in mapping.values()]
        return self.required_inputs_for(targets)

    def dependencies_of(self, target: str | Expr) -> set[str]:
        name = term_name(target)
        if name not in self._terms:
            return set()
        return set(_iter_vars(self._inline(self._terms[name]))) | (self._reachable([name]) - {name})

    def topological_order(self) -> list[str]:
        order: list[str] = []
        seen: set[str] = set()

        def visit(name: str) -> None:
            if name in seen:
                return
            seen.add(name)
            for dep in _iter_vars(self._terms[name]):
                if dep in self._terms:
                    visit(dep)
            order.append(name)

        for name in self._terms:
            visit(name)
        return order

    # ---- introspection ---------------------------------------------------

    def has_view(self, view: str) -> bool:
        return view in self._views

    def view_schema(self, view: str) -> object | None:
        return self._view_schemas.get(view)

    def view_dtypes(self, view: str) -> dict[str, Any | None]:
        return {
            col: (self._meta[node].dtype if node in self._meta else None)
            for col, node in self._views[view].items()
        }

    def formula_of(self, target: str | Expr) -> str:
        name = term_name(target)
        if name not in self._terms:
            return name
        lhs = self._meta[name].symbol or name
        return f"{lhs} = {to_latex(self._terms[name])}"

    def formulas(self) -> dict[str, str]:
        return {name: self.formula_of(name) for name in self._terms}

    def describe(self, target: str | Expr) -> str:
        name = term_name(target)
        meta = self._meta.get(name)
        if meta is None:
            return f"{name} [input]"
        deps = ", ".join(sorted(d for d in _iter_vars(self._terms[name]))) or "-"
        body = f"{name} tags={meta.tags or '()'}\n  deps: {deps}\n  {meta.description or ''}"
        return body.rstrip()

    def info(self, *, view: str | None = None) -> GraphInfo:
        view_nodes = self._resolve_view(None, view) if view else {}
        selected = set(view_nodes.values())
        formula_nodes = tuple(self.topological_order())
        return GraphInfo(
            name=self.name,
            required_inputs=tuple(sorted(self.required_inputs(view))),
            formula_nodes=formula_nodes,
            intermediate_nodes=tuple(n for n in formula_nodes if n not in selected),
            view_nodes=view_nodes,
            view_dtypes=self.view_dtypes(view) if view else {},
        )

    def edges(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for name, expr in self._terms.items():
            for dep in dict.fromkeys(_iter_vars(expr)):
                out.append((dep, name))
        return out

    def to_mermaid(self, *, math_labels: bool = False, view: str | None = None) -> str:
        lines = ["flowchart LR"]
        for a, b in self.edges():
            if math_labels:
                la, lb = self._label(a), self._label(b)
                lines.append(f'    {a}["{la}"] --> {b}["{lb}"]')
            else:
                lines.append(f"    {a} --> {b}")
        return "\n".join(lines)

    def _label(self, name: str) -> str:
        return self.formula_of(name).replace('"', "'") if name in self._terms else name

    def explain(self, target: str | None = None, *, view: str | None = None) -> str:
        if target is None and view is None:
            raise ValueError("pass target=... or view=...")
        if target is not None:
            targets = [target]
        else:
            assert view is not None
            targets = list(self._views[view].values())
        reachable = self._reachable(targets)
        path = [n for n in self.topological_order() if n in reachable]
        lines = [f"FormulaGraph {self.name}"]
        if view is not None:
            schema = self._view_schemas.get(view)
            sname = getattr(schema, "__name__", None)
            lines += ["", "View:", f"  - {view}" + (f" -> {sname}" if sname else "")]
        lines += ["", "Inputs: " + (", ".join(sorted(self.required_inputs(view))) or "-")]
        lines += ["", "Formulas:"]
        for name in path:
            desc = self._meta[name].description
            suffix = f"  # {desc}" if desc else ""
            lines.append(f"  - {self.formula_of(name)}{suffix}")
        if view is not None:
            lines += ["", "Returns:"]
            for col, tname in self._views[view].items():
                lines.append(f"  - {col} <- {tname}")
        return "\n".join(lines)

    def validate_view(self, view: str, schema: object) -> None:
        mapping = self._resolve_view(None, view)
        fields = set(cast(Any, schema).to_schema().columns.keys())
        missing = sorted(fields - set(mapping))
        if missing:
            raise ValueError(f"view {view!r} is missing schema fields: {missing}")


# ---- typed contract-oriented facade ------------------------------------------

_OUTPUT_VIEW = "output"


class _PricingGraphFactory:
    """What ``Formula[Contract, Output]`` evaluates to: a callable remembering the
    two type arguments and building the graph from a name."""

    __slots__ = ("_contract", "_output")

    def __init__(self, contract: Any, output: Any) -> None:
        self._contract = contract
        self._output = output

    def __call__(self, name: str) -> Formula:
        return Formula(name, contract=self._contract, output=self._output)


class Formula:
    """A pure, contract-oriented pricing graph over a pair of boundary schemas.

    ``Formula[Contract, Output](name)`` builds a graph whose inputs are
    ``Contract`` columns (contract *and* pre-resolved market columns alike) and
    whose primary result satisfies ``Output``. Market binding happens *before* the
    graph, in :func:`schenberg.market_data.roles.bind`; the graph is a pure
    function of its input frame.
    """

    def __init__(
        self,
        name: str,
        *,
        contract: type[Any] | None = None,
        output: type[Any] | None = None,
    ) -> None:
        self.name = name
        self._contract = contract
        self._output = output
        self._g = FormulaGraph(name, input=contract)
        self._contract_names = (
            set(cast(Any, contract).to_schema().columns.keys()) if contract is not None else set()
        )

    def __class_getitem__(cls, item: tuple[Any, Any]) -> _PricingGraphFactory:
        contract, output = item
        return _PricingGraphFactory(contract, output)

    @property
    def input(self) -> _InputNS:
        return self._g.input

    @property
    def contract(self) -> _InputNS:
        return self._g.input

    def let(self, name: str, expr: Expr | float | int, **meta: Any) -> Expr:
        return self._g.let(name, expr, **meta)

    def formula(
        self, **meta: Any
    ) -> Callable[[Callable[..., Expr | float | int]], Callable[..., Expr | float | int]]:
        return self._g.formula(**meta)

    def returns(
        self,
        view_or_schema: str | type[Any] | None = None,
        schema: type[Any] | None = None,
        /,
        **mapping: object,
    ) -> Formula:
        if isinstance(view_or_schema, str):
            view = view_or_schema
            resolved = schema if schema is not None else self._output
            if resolved is None:
                raise ValueError(f"graph {self.name!r} has no output schema")
            self._output = resolved
            if mapping:
                self._g.returns(view, resolved, **mapping)
            else:
                self.view(view, resolved)
            return self
        resolved = view_or_schema if view_or_schema is not None else (schema or self._output)
        if resolved is None:
            raise ValueError(f"graph {self.name!r} has no output schema")
        self._output = resolved
        return self.view(_OUTPUT_VIEW, resolved)

    def view(self, name: str, schema: type[Any]) -> Formula:
        fields = list(cast(Any, schema).to_schema().columns.keys())
        # Pass-through contract/input columns need no term; returns() maps by name.
        unmapped = {f: f for f in fields if f not in self._g._terms}
        self._g.returns(name, schema, **{f: t for f, t in unmapped.items()})
        return self

    def plan(self, frame: pl.LazyFrame, *, view: str = _OUTPUT_VIEW) -> pl.LazyFrame:
        """The pure lazy plan over an already-bound frame, projected to the view's
        schema. This is also the :class:`~schenberg.core.router.Computation`
        interface a router branch satisfies. Stays lazy."""
        planned = self._g.plan(frame, view=view)
        schema = self._g.view_schema(view)
        if schema is None:
            return planned
        return planned.select(list(cast(Any, schema).to_schema().columns.keys()))

    def has_view(self, view: str) -> bool:
        return self._g.has_view(view)

    def view_schema(self, view: str) -> object | None:
        return self._g.view_schema(view)

    # ---- introspection passthroughs --------------------------------------
    def explain(self, *, view: str = _OUTPUT_VIEW, **kwargs: Any) -> str:
        return self._g.explain(view=view, **kwargs)

    def to_mermaid(self, *, view: str = _OUTPUT_VIEW, **kwargs: Any) -> str:
        return self._g.to_mermaid(view=view, **kwargs)

    def to_latex(self, target: str | Expr) -> str:
        return self._g.formula_of(target)

    def info(self, *, view: str = _OUTPUT_VIEW) -> GraphInfo:
        return self._g.info(view=view)

    def topological_order(self) -> list[str]:
        return self._g.topological_order()

    def required_inputs(self, view: str = _OUTPUT_VIEW) -> set[str]:
        return self._g.required_inputs(view)

    def dependencies_of(self, target: str | Expr) -> set[str]:
        return self._g.dependencies_of(target)
