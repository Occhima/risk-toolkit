"""Pricing execution contexts for distributed Schenberg graph computations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import polars as pl
from schenberg.core.graph import FormulaGraph

from .backends import PricingBackend, collect_custom, collect_local, collect_ray


@dataclass(frozen=True, slots=True)
class PricingExecutionContext:
    """Materialization policy for lazy pricing computations.

    Schenberg pricing graphs intentionally return ``polars.LazyFrame`` objects.
    This context describes how the final ``collect`` boundary should run,
    including any Polars ``LazyFrame.collect`` keyword arguments and optional
    distributed backend configuration.
    """

    backend: PricingBackend = PricingBackend.LOCAL
    collect_kwargs: dict[str, Any] = field(default_factory=dict)
    ray_init_kwargs: dict[str, Any] = field(default_factory=dict)
    custom_backend: str | None = None

    @classmethod
    def local(cls, **collect_kwargs: Any) -> PricingExecutionContext:
        """Create a local Polars collect context."""
        return cls(backend=PricingBackend.LOCAL, collect_kwargs=dict(collect_kwargs))

    @classmethod
    def ray(
        cls,
        *,
        init_kwargs: Mapping[str, Any] | None = None,
        **collect_kwargs: Any,
    ) -> PricingExecutionContext:
        """Create a Ray-aware collect context.

        ``collect_kwargs`` are forwarded to ``polars.LazyFrame.collect``.
        ``init_kwargs`` are forwarded to ``ray.init`` when Ray is not already
        initialized.
        """
        return cls(
            backend=PricingBackend.RAY,
            collect_kwargs=dict(collect_kwargs),
            ray_init_kwargs=dict(init_kwargs or {}),
        )

    @classmethod
    def custom(cls, name: str, **collect_kwargs: Any) -> PricingExecutionContext:
        """Create a context for a registered custom backend."""
        return cls(
            backend=PricingBackend.CUSTOM,
            collect_kwargs=dict(collect_kwargs),
            custom_backend=name,
        )

    def collect(self, lf: pl.LazyFrame) -> pl.DataFrame:
        """Materialize a lazy pricing frame with this context."""
        if self.backend is PricingBackend.LOCAL:
            return collect_local(lf, self.collect_kwargs)
        if self.backend is PricingBackend.RAY:
            return collect_ray(lf, self.collect_kwargs, init_kwargs=self.ray_init_kwargs)
        if self.backend is PricingBackend.CUSTOM:
            if self.custom_backend is None:
                raise ValueError("custom backend contexts require custom_backend")
            return collect_custom(lf, self.collect_kwargs, name=self.custom_backend)
        raise ValueError(f"unsupported pricing backend {self.backend!r}")


def collect_pricing(
    lf: pl.LazyFrame,
    *,
    context: PricingExecutionContext | None = None,
) -> pl.DataFrame:
    """Collect an already-built lazy pricing frame using a pricing context."""
    return (context or PricingExecutionContext.local()).collect(lf)


def compute_graph_pricing(
    graph: FormulaGraph,
    lf: pl.LazyFrame,
    *,
    context: PricingExecutionContext | None = None,
    outputs: Mapping[str, str] | None = None,
    view: str | None = None,
) -> pl.DataFrame:
    """Plan and collect graph pricing outputs with a selected backend. The frame
    must already be market-bound — the graph is a pure function of its inputs."""
    pricing_lf = graph.plan(lf, outputs=outputs, view=view)
    return collect_pricing(pricing_lf, context=context)


def stage_graph_pricing(
    graph: FormulaGraph,
    lf: pl.LazyFrame,
    *,
    context: PricingExecutionContext | None = None,
    view: str | None = None,
    targets: list[str] | None = None,
) -> pl.DataFrame:
    """Materialize staged graph intermediates with a selected backend."""
    staged_lf = graph.stage(lf, view=view, targets=targets)
    return collect_pricing(staged_lf, context=context)
