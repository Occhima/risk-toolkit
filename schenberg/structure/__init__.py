"""Minimal structured products as leg composition over priced instruments."""

from __future__ import annotations

import polars as pl

from schenberg.core.fold import Fold, strict_sum_
from schenberg.domain.schemas.position import InstrumentValue
from schenberg.domain.schemas.structure import StructureLeg, StructureLegValue, StructureValue


def structure_stage(
    legs: pl.LazyFrame,
    values: pl.LazyFrame,
    *,
    on: tuple[str, ...] = ("instrument_type", "instrument_id"),
) -> pl.LazyFrame:
    """Join priced legs and expose per-leg weighted values for debugging."""
    value_cols = list(InstrumentValue.to_schema().columns.keys())
    return legs.join(values.select(value_cols), on=list(on), how="left").with_columns(
        (pl.col("quantity") * pl.col("weight") * pl.col("value")).alias("leg_value")
    )


structure_value_fold = (
    Fold("structure_value", input_schema=StructureLegValue)
    .by("structure_id")
    .returns(
        StructureValue,
        value=strict_sum_("leg_value"),
    )
)


def structure_value(
    legs: pl.LazyFrame,
    values: pl.LazyFrame,
    *,
    on: tuple[str, ...] = ("instrument_type", "instrument_id"),
) -> pl.LazyFrame:
    """Aggregate structure leg values with the inspectable structure fold."""
    return structure_value_fold.compute(structure_stage(legs, values, on=on))


__all__ = [
    "StructureLeg",
    "StructureLegValue",
    "StructureValue",
    "structure_stage",
    "structure_value",
    "structure_value_fold",
]
