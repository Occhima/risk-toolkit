"""Minimal structured products as leg composition over priced instruments."""

from __future__ import annotations

import polars as pl

from schenberg.domain.schemas.position import InstrumentValue
from schenberg.domain.schemas.structure import StructureLeg


def structure_value(
    legs: pl.LazyFrame,
    values: pl.LazyFrame,
    *,
    on: tuple[str, ...] = ("instrument_type", "instrument_id"),
) -> pl.LazyFrame:
    """Join priced legs and sum ``quantity * weight * value`` by structure."""
    value_cols = list(InstrumentValue.to_schema().columns.keys())
    return (
        legs.join(values.select(value_cols), on=list(on), how="left")
        .with_columns((pl.col("quantity") * pl.col("weight") * pl.col("value")).alias("leg_value"))
        .group_by("structure_id")
        .agg(
            pl.when(pl.col("leg_value").null_count() > 0)
            .then(None)
            .otherwise(pl.sum("leg_value"))
            .alias("value")
        )
    )


__all__ = ["StructureLeg", "structure_value"]
