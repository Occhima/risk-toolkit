"""Backend registry for Schenberg pricing materialization."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Any, cast

import polars as pl
import ray

CollectFn = Callable[[pl.LazyFrame, Mapping[str, Any]], pl.DataFrame]


class PricingBackend(StrEnum):
    """Supported pricing execution backends.

    The Schenberg core pricing package stays lazy. Backends in this plugin are
    responsible only for the final materialization boundary.
    """

    LOCAL = "local"
    RAY = "ray"
    CUSTOM = "custom"


_CUSTOM_BACKENDS: dict[str, CollectFn] = {}


def register_backend(name: str, collect_fn: CollectFn) -> None:
    """Register a custom materialization backend.

    Custom backends receive the lazy pricing frame plus the normalized Polars
    ``collect`` keyword arguments from a pricing context.
    """
    if not name:
        raise ValueError("backend name must not be empty")
    _CUSTOM_BACKENDS[name] = collect_fn


def collect_local(lf: pl.LazyFrame, collect_kwargs: Mapping[str, Any]) -> pl.DataFrame:
    """Collect a lazy pricing frame with Polars' local engine."""
    return cast(pl.DataFrame, lf.collect(**dict(collect_kwargs)))


def collect_ray(
    lf: pl.LazyFrame,
    collect_kwargs: Mapping[str, Any],
    *,
    init_kwargs: Mapping[str, Any] | None = None,
) -> pl.DataFrame:
    """Collect a lazy pricing frame after ensuring Ray is initialized."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, **dict(init_kwargs or {}))
    return cast(pl.DataFrame, lf.collect(**dict(collect_kwargs)))


def collect_custom(
    lf: pl.LazyFrame,
    collect_kwargs: Mapping[str, Any],
    *,
    name: str,
) -> pl.DataFrame:
    """Collect using a registered third-party backend hook."""
    try:
        collect_fn = _CUSTOM_BACKENDS[name]
    except KeyError:
        available = ", ".join(sorted(_CUSTOM_BACKENDS)) or "none"
        msg = f"unknown custom backend {name!r}; available backends: {available}"
        raise KeyError(msg) from None
    return collect_fn(lf, collect_kwargs)
