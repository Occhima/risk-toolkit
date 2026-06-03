"""Formula-graph engine.

A semantic formula DAG (backed by rustworkx) that compiles into Polars
expressions. rustworkx is an internal detail: it never appears in a public
signature. Nothing in this module calls .collect().
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from inspect import signature
from typing import Any, cast

import polars as pl
import rustworkx as rx

from .market import MarketRequirement, MarketSnapshot

ExprFn = Callable[..., pl.Expr]


class NodeKind(StrEnum):
    INPUT = "input"  # a column expected to already exist in the frame (fn is None)
    FORMULA = "formula"  # derived from other nodes via fn


@dataclass(frozen=True, slots=True)
class ExprNode:
    name: str
    kind: NodeKind
    deps: tuple[str, ...] = ()
    fn: ExprFn | None = None
    dtype: Any | None = None
    tags: tuple[str, ...] = ()
    description: str | None = None


class ExprGraph:
    """A semantic formula DAG that compiles into Polars expressions.

    rustworkx owns the topology (cycle checks, topo order, ancestors); Polars
    owns execution. Edge direction is dependency -> dependent.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._graph: rx.PyDiGraph = rx.PyDiGraph(multigraph=False)
        self._indices: dict[str, int] = {}  # node name -> rustworkx index
        self._input_aliases: dict[str, str] = {}  # dep name -> concrete node name
        self._output_profiles: dict[str, dict[str, str]] = {}  # profile -> {out_col: node_name}
        self._market: list[MarketRequirement] = []

    # ---- construction ----------------------------------------------------

    def node(
        self,
        *,
        dtype: Any = pl.Float64,
        tags: tuple[str, ...] = (),
        description: str | None = None,
        name: str | None = None,
    ):
        """Decorator. Dependencies are inferred from the parameter names."""

        def register(fn: ExprFn) -> ExprFn:
            self._register(
                ExprNode(
                    name=name or cast(str, getattr(fn, "__name__", None)),
                    kind=NodeKind.FORMULA,
                    deps=tuple(signature(fn).parameters),
                    fn=fn,
                    dtype=dtype,
                    tags=tuple(tags),
                    description=description,
                )
            )
            return fn

        return register

    def _register(self, node: ExprNode) -> None:
        if node.name in self._indices:
            raise ValueError(f"node {node.name!r} already defined in graph {self.name!r}")
        idx = self._graph.add_node(node)
        self._indices[node.name] = idx
        for dep in node.deps:
            self._graph.add_edge(self._ensure_input(dep), idx, None)  # dep -> dependent
        self._validate_dag()

    def _ensure_input(self, name: str) -> int:
        idx = self._indices.get(name)
        if idx is None:
            idx = self._graph.add_node(ExprNode(name, NodeKind.INPUT))
            self._indices[name] = idx
        return idx

    def _validate_dag(self) -> None:
        if not rx.is_directed_acyclic_graph(self._graph):
            raise ValueError(f"cycle detected in graph {self.name!r}")

    # ---- configuration (chainable) ---------------------------------------

    def with_market(self, *requirements: MarketRequirement) -> ExprGraph:
        self._market.extend(requirements)
        return self

    def with_inputs(self, **aliases: str) -> ExprGraph:
        """Redirect a dependency name to a concrete node, e.g.
        with_inputs(signed_cashflow="cdi_signed_cashflow")."""
        self._input_aliases.update(aliases)
        return self

    def with_outputs(
        self,
        profile: str,
        schema: object | None = None,
        /,
        **overrides: str,
    ) -> ExprGraph:
        """Name a set of output columns: out_column -> node_name.

        Two forms:
          with_outputs("pricing", year_fraction="year_fraction", pv="leg_pv")
          with_outputs("pricing", LegPricing, pv="leg_pv")   # schema-driven

        With a schema (any object exposing .to_schema().columns.keys(), e.g. a
        Pandera DataFrameModel), each field maps identity to a node of the same
        name; pass overrides only where the feeding node name differs. The
        schema is duck-typed so the engine stays free of a Pandera dependency.
        """
        if schema is not None:
            typed_schema = cast(Any, schema)
            fields = list(typed_schema.to_schema().columns.keys())
            mapping = {f: overrides.get(f, f) for f in fields}
        else:
            mapping = dict(overrides)
        self._output_profiles[profile] = mapping
        return self

    # ---- composition -----------------------------------------------------

    @classmethod
    def compose(cls, name: str, *graphs: ExprGraph) -> ExprGraph:
        """Merge several graphs into a new one (inputs do not mutate).

        Two-pass build so registration order never matters: collect formulas,
        then create input nodes for unresolved deps, then wire edges. An INPUT
        in one graph satisfied by a FORMULA in another resolves correctly
        because INPUT nodes are recreated, not collected. Two different
        formulas under the same name is a hard conflict.
        """
        merged = cls(name)
        formulas: dict[str, ExprNode] = {}
        for g in graphs:
            for idx in g._indices.values():
                node = g._graph[idx]
                if node.kind is not NodeKind.FORMULA:
                    continue
                prev = formulas.get(node.name)
                if prev is not None and prev != node:
                    raise ValueError(f"conflicting formula {node.name!r} in compose({name!r})")
                formulas[node.name] = node
            merged._input_aliases.update(g._input_aliases)
            merged._market.extend(g._market)

        for node in formulas.values():  # 1. formula nodes
            merged._indices[node.name] = merged._graph.add_node(node)
        for node in formulas.values():  # 2. inputs for unresolved deps
            for dep in node.deps:
                merged._ensure_input(dep)
        for node in formulas.values():  # 3. edges dep -> dependent
            for dep in node.deps:
                merged._graph.add_edge(merged._indices[dep], merged._indices[node.name], None)
        merged._validate_dag()
        return merged

    # ---- compilation -----------------------------------------------------

    def _resolve(self, name: str) -> str:
        """Follow the input-alias chain to the concrete node name."""
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
        """Recursively compile `target` into one nested pl.Expr."""
        cache = _cache if _cache is not None else {}
        if target in cache:
            return cache[target]

        # 1. explicit override
        if provided and target in provided:
            return cache.setdefault(target, provided[target])

        # 2. input alias (signed_cashflow -> cdi_signed_cashflow)
        alias = self._input_aliases.get(target)
        if alias is not None and alias != target:
            return cache.setdefault(target, self.expr(alias, provided, cache))

        # 3 + 4. resolve node
        idx = self._indices.get(target)
        if idx is None:
            raise KeyError(
                f"{target!r} is not a node in graph {self.name!r}; "
                f"known nodes: {sorted(self._indices)}"
            )
        node = self._graph[idx]
        if node.fn is None:  # input column
            result = pl.col(target)
        else:  # formula
            args = [self.expr(dep, provided, cache) for dep in node.deps]
            result = node.fn(*args)
        cache[target] = result
        return result

    def compute_for(
        self,
        lf: pl.LazyFrame,
        *,
        market: MarketSnapshot | None = None,
        outputs: Mapping[str, str] | None = None,
        output_profile: str | None = None,
    ) -> pl.LazyFrame:
        """Attach market data (if required), compile each output to a nested
        expression, add them with one with_columns. Stays lazy."""
        lf = self._attach_market(lf, market)
        mapping = self._resolve_outputs(outputs, output_profile)
        cache: dict[str, pl.Expr] = {}  # share intermediates across outputs
        columns = [self.expr(node, _cache=cache).alias(col) for col, node in mapping.items()]
        return lf.with_columns(columns)

    def stage(
        self,
        lf: pl.LazyFrame,
        *,
        market: MarketSnapshot | None = None,
        output_profile: str | None = None,
        targets: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Analysis mode: materialize every intermediate node as its own column.
        Wider/slower than compute_for, but every step is inspectable. Stays lazy.

        Debug workflow: graph.stage(...).collect().null_count() -- nulls
        propagate, so the first column that is unexpectedly null is the root
        cause (e.g. null zero_rate => failed curve join).
        """
        lf = self._attach_market(lf, market)
        if targets is None:
            if output_profile is None:
                raise ValueError("pass targets=[...] or output_profile=...")
            targets = list(self._output_profiles[output_profile].values())

        order: list[str] = []
        seen: set[str] = set()

        def emit(name: str) -> None:  # post-order over RESOLVED deps
            real = self._resolve(name)
            if real in seen:
                return
            seen.add(real)
            node = self._graph[self._indices[real]]
            if node.fn is None:  # input column, already present
                return
            for dep in node.deps:
                emit(dep)
            order.append(real)

        for t in targets:
            emit(t)

        for name in order:
            node = self._graph[self._indices[name]]
            cols = [pl.col(self._resolve(d)) for d in node.deps]
            lf = lf.with_columns(node.fn(*cols).alias(name))
        return lf

    def _attach_market(self, lf: pl.LazyFrame, market: MarketSnapshot | None) -> pl.LazyFrame:
        if self._market:
            if market is None:
                raise ValueError(f"graph {self.name!r} needs a MarketSnapshot")
            for req in self._market:
                lf = market.attach(lf, req)
        return lf

    def _resolve_outputs(
        self, outputs: Mapping[str, str] | None, profile: str | None
    ) -> dict[str, str]:
        if outputs is not None:
            return dict(outputs)
        if profile is not None:
            try:
                return dict(self._output_profiles[profile])
            except KeyError:
                raise KeyError(f"no output profile {profile!r} in graph {self.name!r}") from None
        raise ValueError("pass outputs={...} or output_profile=...")

    # ---- introspection ---------------------------------------------------

    def edges(self) -> list[tuple[str, str]]:
        names = {i: self._graph[i].name for i in self._indices.values()}
        return [(names[a], names[b]) for a, b in self._graph.edge_list()]

    def dependencies_of(self, target: str) -> set[str]:
        idx = self._indices[target]
        return {self._graph[i].name for i in rx.ancestors(self._graph, idx)}

    def topological_order(self) -> list[str]:
        return [self._graph[i].name for i in rx.topological_sort(self._graph)]

    def describe(self, target: str) -> str:
        node = self._graph[self._indices[target]]
        return (
            f"{node.name} [{node.kind}] tags={node.tags or '()'}\n"
            f"  deps: {', '.join(node.deps) or '-'}\n"
            f"  {node.description or ''}"
        ).rstrip()

    def to_mermaid(self) -> str:
        lines = ["flowchart LR", *(f"    {a} --> {b}" for a, b in self.edges())]
        return "\n".join(lines)

    def required_inputs(self) -> set[str]:
        """The columns a caller must supply: leaf INPUT nodes, minus what market
        requirements provide, minus aliased dep names, plus market join keys.
        Derived from the graph itself, so it cannot drift from the formulas."""
        graph_inputs = {n for n, i in self._indices.items() if self._graph[i].fn is None}
        market_provided = {out for req in self._market for out in req.outputs.values()}
        market_keys = {k for req in self._market for k in req.left_keys}
        return (graph_inputs - market_provided - set(self._input_aliases)) | market_keys

    def output_dtypes(self, profile: str) -> dict[str, Any | None]:
        """{output_column: declared node dtype} for a profile — a cheap contract
        derived from the stored node dtypes."""
        return {
            col: self._graph[self._indices[node]].dtype
            for col, node in self._output_profiles[profile].items()
        }



def __getattr__(name: str) -> Any:
    if name == "Router":
        from schenberg.core.router import Router  # noqa: PLC0415

        return Router
    raise AttributeError(name)
