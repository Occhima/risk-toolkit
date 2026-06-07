"""Valuation DAG plans for Schenberg pricing nodes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import rustworkx as rx

NodeKind = Literal["input", "function", "concat"]


@dataclass(frozen=True, slots=True)
class ValuationNode:
    """Metadata for one valuation-plan vertex."""

    name: str
    kind: NodeKind
    fn: Callable[..., Any] | None = None
    bindings: Mapping[str, str] = field(default_factory=dict)
    concat_inputs: tuple[str, ...] = ()
    concat_how: str = "diagonal_relaxed"
    market_sources: tuple[str, ...] = ()
    description: str | None = None


class ValuationPlan:
    """Small rustworkx-backed DAG of whole valuation nodes.

    Edges point from dependency to dependent, e.g. ``trades -> forward_values``.
    Building the plan records callables only; decorated functions are not executed
    until an executor runs a target.
    """

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("valuation plan name must not be empty")
        self.name = name
        self._graph = rx.PyDiGraph()
        self._indices: dict[str, int] = {}
        self._inputs: dict[str, Any] = {}
        self._nodes: dict[str, ValuationNode] = {}

    def input(self, name: str, value: Any) -> ValuationPlan:
        """Register an input value and return this plan."""
        self._ensure_available(name)
        node = ValuationNode(name=name, kind="input")
        idx = self._graph.add_node(node)
        self._indices[name] = idx
        self._inputs[name] = value
        return self

    def node(
        self,
        name: str | None = None,
        *,
        market_sources: tuple[str, ...] = (),
        description: str | None = None,
        **bindings: str,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorate and register a function node.

        ``bindings`` maps function argument names to dependency names in this
        plan. The original function is returned unchanged.
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            node_name = name or getattr(fn, "__name__", "")
            if not node_name:
                raise ValueError("node name must not be empty")
            self._ensure_available(node_name)
            self._ensure_dependencies(bindings.values(), node_name=node_name)
            vnode = ValuationNode(
                name=node_name,
                kind="function",
                fn=fn,
                bindings=dict(bindings),
                market_sources=tuple(market_sources),
                description=description,
            )
            self._add_node(vnode, dependencies=tuple(bindings.values()))
            return fn

        return decorator

    def concat(
        self,
        name: str,
        inputs: tuple[str, ...] | list[str],
        *,
        how: str = "diagonal_relaxed",
        description: str | None = None,
    ) -> ValuationPlan:
        """Register a concat node that combines whole node outputs."""
        self._ensure_available(name)
        concat_inputs = tuple(inputs)
        if not concat_inputs:
            raise ValueError(f"concat node {name!r} requires at least one input")
        self._ensure_dependencies(concat_inputs, node_name=name)
        vnode = ValuationNode(
            name=name,
            kind="concat",
            concat_inputs=concat_inputs,
            concat_how=how,
            description=description,
        )
        self._add_node(vnode, dependencies=concat_inputs)
        return self

    def has(self, name: str) -> bool:
        return name in self._indices

    def get(self, name: str) -> ValuationNode:
        self._require_name(name)
        if name in self._nodes:
            return self._nodes[name]
        return self._graph[self._indices[name]]

    def inputs(self) -> tuple[str, ...]:
        return tuple(self._inputs)

    def nodes(self) -> tuple[str, ...]:
        return tuple(self._nodes)

    def depends_on(self, name: str) -> tuple[str, ...]:
        self._require_name(name)
        idx = self._indices[name]
        return tuple(self._graph[i].name for i in self._graph.predecessor_indices(idx))

    def downstream_of(self, name: str) -> tuple[str, ...]:
        self._require_name(name)
        indices = set(rx.descendants(self._graph, self._indices[name]))
        return self._ordered_names(indices)

    def upstream_of(self, name: str) -> tuple[str, ...]:
        self._require_name(name)
        indices = set(rx.ancestors(self._graph, self._indices[name]))
        return self._ordered_names(indices)

    def transitive_dependencies(self, name: str) -> tuple[str, ...]:
        return self.upstream_of(name)

    def topological_order(self, target: str | None = None) -> tuple[str, ...]:
        try:
            order = tuple(rx.topological_sort(self._graph))
        except rx.DAGHasCycle as exc:
            raise ValueError(f"valuation plan {self.name!r} contains a cycle") from exc
        if target is None:
            return tuple(self._graph[i].name for i in order)
        self._require_name(target)
        required = set(rx.ancestors(self._graph, self._indices[target])) | {self._indices[target]}
        return tuple(self._graph[i].name for i in order if i in required)

    def affected_by_market_source(self, source: str) -> tuple[str, ...]:
        starts = {
            self._indices[name]
            for name, node in self._nodes.items()
            if source in node.market_sources
        }
        affected = set(starts)
        for idx in starts:
            affected.update(rx.descendants(self._graph, idx))
        return self._ordered_names(affected)

    def validate(self) -> None:
        self.topological_order()
        for name, node in self._nodes.items():
            if node.kind == "function":
                if node.fn is None:
                    raise ValueError(f"function node {name!r} has no callable")
                self._ensure_dependencies(node.bindings.values(), node_name=name)
            elif node.kind == "concat":
                self._ensure_dependencies(node.concat_inputs, node_name=name)

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inputs": list(self.inputs()),
            "nodes": [
                {
                    "name": node.name,
                    "kind": node.kind,
                    "dependencies": list(self.depends_on(node.name)),
                    "market_sources": list(node.market_sources),
                    "description": node.description,
                }
                for node in (self._nodes[name] for name in self.nodes())
            ],
        }

    def explain(self) -> str:
        lines = [f"ValuationPlan {self.name}", "", "Inputs:"]
        lines += [f"  - {name}" for name in self.inputs()] or ["  - (none)"]
        lines += ["", "Nodes:"]
        if not self._nodes:
            lines.append("  - (none)")
        for name in self.nodes():
            node = self._nodes[name]
            lines.append(f"  - {name} [{node.kind}]")
            deps = ", ".join(self.depends_on(name)) or "-"
            lines.append(f"    deps: {deps}")
            if node.kind == "concat":
                lines.append(f"    concat_how: {node.concat_how}")
            if node.market_sources:
                lines.append(f"    market_sources: {', '.join(node.market_sources)}")
            if node.description:
                lines.append(f"    description: {node.description}")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        lines = ["flowchart LR"]
        for name in self.inputs():
            lines.append(f'    {self._mermaid_id(name)}["input: {self._escape(name)}"]')
        for name in self.nodes():
            node = self._nodes[name]
            label = f"{name} ({node.kind})" if node.kind == "concat" else name
            lines.append(f'    {self._mermaid_id(name)}["{self._escape(label)}"]')
        for name in self.topological_order():
            for dep in self.depends_on(name):
                lines.append(f"    {self._mermaid_id(dep)} --> {self._mermaid_id(name)}")
        return "\n".join(lines)

    def _add_node(self, node: ValuationNode, *, dependencies: tuple[str, ...]) -> None:
        idx = self._graph.add_node(node)
        self._indices[node.name] = idx
        self._nodes[node.name] = node
        for dep in dependencies:
            self._graph.add_edge(self._indices[dep], idx, None)

    def _ensure_available(self, name: str) -> None:
        if not name:
            raise ValueError("name must not be empty")
        if name in self._indices:
            raise ValueError(f"valuation plan {self.name!r} already has {name!r}")

    def _ensure_dependencies(self, dependencies: Any, *, node_name: str) -> None:
        missing = [dep for dep in dependencies if dep not in self._indices]
        if missing:
            raise ValueError(
                f"node {node_name!r} references missing dependency/dependencies: {missing}"
            )

    def _require_name(self, name: str) -> None:
        if name not in self._indices:
            raise KeyError(f"valuation plan {self.name!r} has no node or input {name!r}")

    def _ordered_names(self, indices: set[int]) -> tuple[str, ...]:
        return tuple(name for name in self.topological_order() if self._indices[name] in indices)

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace('"', "'")

    @staticmethod
    def _mermaid_id(value: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value)
        if not safe or safe[0].isdigit():
            safe = f"n_{safe}"
        return safe
