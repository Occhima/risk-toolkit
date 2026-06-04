"""Workflow: a DAG of STAGES (the workflow layer).

Same composition idea as FormulaGraph (decorator + signature-inferred deps +
rustworkx topo order), but nodes are stage functions returning LazyFrames, not
pl.Expr. Use it when shapes change between steps — joins, group_by, repricing
under a bumped market — i.e. things that can never be a single column expression.

A stage's parameter names are its dependencies: either other stages or external
inputs passed to .run(). Nothing collects.
"""

from __future__ import annotations

from collections.abc import Callable
from inspect import signature
from typing import Any

import rustworkx as rx


class Workflow:
    def __init__(self, name: str) -> None:
        self.name = name
        self._fns: dict[str, tuple[Callable[..., Any], tuple[str, ...]]] = {}
        self._order: list[str] | None = None

    def stage(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator. Register a stage; its parameter names are its deps."""
        self._fns[fn.__name__] = (fn, tuple(signature(fn).parameters))  # ty: ignore[unresolved-attribute]
        self._order = None
        return fn

    def _compile(self) -> None:
        if self._order is not None:
            return
        g, idx = rx.PyDiGraph(), {}
        for nm in self._fns:
            idx[nm] = g.add_node(nm)
        for nm, (_, deps) in self._fns.items():
            for dep in deps:
                if dep in idx:  # dep is another stage -> edge
                    g.add_edge(idx[dep], idx[nm], None)
        if not rx.is_directed_acyclic_graph(g):
            raise ValueError(f"workflow {self.name!r} has a cycle")
        self._order = [g[i] for i in rx.topological_sort(g)]

    def run(self, **inputs: Any) -> dict[str, Any]:
        """Execute stages in topological order. Returns a dict of every external
        input plus every stage output, keyed by name. Stays lazy — stage outputs
        are LazyFrames; collect only what you need afterwards."""
        self._compile()
        env: dict[str, Any] = dict(inputs)
        assert self._order is not None
        for nm in self._order:
            fn, deps = self._fns[nm]
            env[nm] = fn(*(env[d] for d in deps))
        return env

    def order(self) -> list[str]:
        self._compile()
        assert self._order is not None
        return list(self._order)

    # ---- introspection ---------------------------------------------------

    def stages(self) -> list[str]:
        """The registered stage names (declaration order)."""
        return list(self._fns)

    def dependencies(self) -> dict[str, list[str]]:
        """Each stage mapped to its dependency names (other stages or external
        inputs supplied to :meth:`run`)."""
        return {name: list(deps) for name, (_, deps) in self._fns.items()}

    def external_inputs(self) -> list[str]:
        """Dependency names that are not themselves stages — the inputs a caller
        must pass to :meth:`run`."""
        stage_names = set(self._fns)
        seen: dict[str, None] = {}
        for _, deps in self._fns.values():
            for dep in deps:
                if dep not in stage_names:
                    seen.setdefault(dep, None)
        return list(seen)

    def info(self) -> dict[str, object]:
        return {
            "name": self.name,
            "stages": self.stages(),
            "order": self.order(),
            "external_inputs": self.external_inputs(),
            "dependencies": self.dependencies(),
        }

    def explain(self) -> str:
        deps = self.dependencies()
        stage_names = set(self._fns)
        lines = [f"Workflow {self.name}", "", "External inputs:"]
        lines += [f"  - {name}" for name in self.external_inputs()] or ["  - (none)"]
        lines += ["", "Stages (topological order):"]
        for name in self.order():
            wired = ", ".join(d for d in deps[name]) or "-"
            lines.append(f"  - {name}({wired})")
        lines += ["", "Edges:"]
        edges = [
            f"  - {dep} -> {name}"
            for name, dnames in deps.items()
            for dep in dnames
            if dep in stage_names
        ]
        lines += edges or ["  - (none)"]
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        deps = self.dependencies()
        stage_names = set(self._fns)
        lines = ["flowchart LR"]
        for name, dnames in deps.items():
            for dep in dnames:
                shape = f"{dep}" if dep in stage_names else f'{dep}(["{dep}"])'
                lines.append(f"    {shape} --> {name}")
        return "\n".join(lines)
