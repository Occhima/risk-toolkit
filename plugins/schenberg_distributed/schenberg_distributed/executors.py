"""Executors for whole-node Schenberg valuation plans."""

from __future__ import annotations

import importlib
from typing import Any, cast

import polars as pl

from .context import PricingExecutionContext
from .plan import ValuationPlan


def _concat_preserve_lazy(frames: list[Any], *, how: str) -> pl.DataFrame | pl.LazyFrame:
    if any(isinstance(frame, pl.LazyFrame) for frame in frames):
        lazy_frames = [
            frame.lazy() if isinstance(frame, pl.DataFrame) else frame for frame in frames
        ]
        return pl.concat(lazy_frames, how=how)
    return pl.concat(frames, how=how)


def _materialize(value: Any) -> Any:
    if isinstance(value, pl.LazyFrame):
        return value.collect()
    return value


def _run_function(fn: Any, kwargs: dict[str, Any]) -> Any:
    return _materialize(fn(**kwargs))


def _run_concat(frames: list[Any], how: str) -> pl.DataFrame:
    normalized = [_materialize(frame) for frame in frames]
    return cast(pl.DataFrame, pl.concat(normalized, how=how))


class LocalExecutor:
    """Synchronous local executor that preserves Polars laziness where possible."""

    def __init__(self, context: PricingExecutionContext | None = None) -> None:
        self.context = context or PricingExecutionContext.local()

    def lazy(self, plan: ValuationPlan, *, target: str) -> Any:
        plan.validate()
        if not plan.has(target):
            raise KeyError(f"valuation plan {plan.name!r} has no target {target!r}")
        env = dict(plan._inputs)
        for name in plan.topological_order(target):
            if name in plan._inputs:
                continue
            node = plan.get(name)
            if node.kind == "function":
                if node.fn is None:
                    raise ValueError(f"function node {name!r} has no callable")
                kwargs = {arg: env[dep] for arg, dep in node.bindings.items()}
                result = node.fn(**kwargs)
            elif node.kind == "concat":
                frames = [env[input_name] for input_name in node.concat_inputs]
                result = _concat_preserve_lazy(frames, how=node.concat_how)
            else:
                raise ValueError(f"cannot execute node {name!r} of kind {node.kind!r}")
            env[name] = result
        return env[target]

    def collect(self, plan: ValuationPlan, *, target: str) -> pl.DataFrame:
        result = self.lazy(plan, target=target)
        if isinstance(result, pl.DataFrame):
            return result
        if isinstance(result, pl.LazyFrame):
            return self.context.collect(result)
        raise TypeError(
            f"target {target!r} returned {type(result).__name__}; "
            "expected Polars DataFrame or LazyFrame"
        )


class DaskExecutor:
    """Optional Dask executor; each valuation node becomes one delayed task."""

    def __init__(self, context: PricingExecutionContext | None = None) -> None:
        self.context = context or PricingExecutionContext.local()

    def collect(self, plan: ValuationPlan, *, target: str) -> pl.DataFrame:
        try:
            delayed = importlib.import_module("dask").delayed
        except ImportError as exc:
            raise ImportError(
                "DaskExecutor requires installing schenberg-distributed[dask]"
            ) from exc

        plan.validate()
        if not plan.has(target):
            raise KeyError(f"valuation plan {plan.name!r} has no target {target!r}")
        env: dict[str, Any] = {
            name: delayed(_materialize)(value) for name, value in plan._inputs.items()
        }
        for name in plan.topological_order(target):
            if name in plan._inputs:
                continue
            node = plan.get(name)
            if node.kind == "function":
                if node.fn is None:
                    raise ValueError(f"function node {name!r} has no callable")
                kwargs = {arg: env[dep] for arg, dep in node.bindings.items()}
                env[name] = delayed(_run_function)(node.fn, kwargs)
            elif node.kind == "concat":
                frames = [env[input_name] for input_name in node.concat_inputs]
                env[name] = delayed(_run_concat)(frames, node.concat_how)
            else:
                raise ValueError(f"cannot execute node {name!r} of kind {node.kind!r}")
        result = env[target].compute()
        if isinstance(result, pl.DataFrame):
            return result
        if isinstance(result, pl.LazyFrame):
            return self.context.collect(result)
        raise TypeError(
            f"target {target!r} returned {type(result).__name__}; "
            "expected Polars DataFrame or LazyFrame"
        )
