"""Tiny lazy scenario repricing helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

from schenberg.market_data.shocks import Shock
from schenberg.market_data.snapshot import MarketSnapshot


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    shock: Shock


@dataclass(frozen=True, slots=True)
class ScenarioSet:
    scenarios: tuple[Scenario, ...]

    @classmethod
    def of(cls, *scenarios: Scenario) -> ScenarioSet:
        return cls(tuple(scenarios))


def _select_values(
    lf: pl.LazyFrame, *, id_columns: tuple[str, ...], value_columns: tuple[str, ...], suffix: str
) -> pl.LazyFrame:
    return lf.select([*id_columns, *[pl.col(c).alias(f"{c}_{suffix}") for c in value_columns]])


def reprice_under(
    pricer: Callable[[pl.LazyFrame, MarketSnapshot], pl.LazyFrame],
    trades: pl.LazyFrame,
    market: MarketSnapshot,
    scenarios: ScenarioSet,
    *,
    id_columns: tuple[str, ...] = ("instrument_id",),
    value_columns: tuple[str, ...] = ("value",),
) -> pl.LazyFrame:
    """Return base/shocked/diff values for each scenario without collecting."""
    base = _select_values(
        pricer(trades, market), id_columns=id_columns, value_columns=value_columns, suffix="base"
    )
    frames: list[pl.LazyFrame] = []
    for scenario in scenarios.scenarios:
        shocked = _select_values(
            pricer(trades, market.apply(scenario.shock)),
            id_columns=id_columns,
            value_columns=value_columns,
            suffix="shocked",
        )
        joined = base.join(shocked, on=list(id_columns), how="left").with_columns(
            pl.lit(scenario.name).alias("scenario"),
            *[
                (pl.col(f"{c}_shocked") - pl.col(f"{c}_base")).alias(f"{c}_diff")
                for c in value_columns
            ],
        )
        frames.append(
            joined.select(
                [
                    *id_columns,
                    "scenario",
                    *[f"{c}_base" for c in value_columns],
                    *[f"{c}_shocked" for c in value_columns],
                    *[f"{c}_diff" for c in value_columns],
                ]
            )
        )
    if not frames:
        return base.with_columns(pl.lit(None, dtype=pl.String).alias("scenario"))
    return pl.concat(frames, how="vertical")
