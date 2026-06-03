"""Implied-volatility surfaces as declarative market data.

Volatility is market data, not a manual preprocessing step in option pricers.
Option graphs declare a :class:`VolSurfaceRequirement` with ``graph.with_market``;
``ExprGraph.compute_for(..., market=...)`` then attaches the implied volatility
before formula compilation. The requirement is attachable market data because it
interpolates a surface instead of performing a simple key join.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandera.polars as pa
import polars as pl
from numpy.typing import ArrayLike, NDArray
from pandera.typing.polars import LazyFrame

from schenberg.domain.schemas.market_data import VolSurfaceContract
from schenberg.market_data.sources import MarketSource
from schenberg.math.interpolation import bilinear

if TYPE_CHECKING:
    from schenberg.market_data.snapshot import MarketSnapshot


def _collect_quotes(
    quotes: pl.LazyFrame | pl.DataFrame | LazyFrame[VolSurfaceContract],
) -> pl.DataFrame:
    if isinstance(quotes, pl.DataFrame):
        return quotes
    return cast(pl.DataFrame, quotes.collect())


@dataclass(frozen=True, slots=True)
class VolSurface:
    """Bilinear implied-vol surface for a single underlying.

    ``vols[i, j]`` is the implied vol at ``(tenors[i], strikes[j])``. Tenors are
    stored as year fractions.
    """

    name: str
    tenors: NDArray[np.float64]
    strikes: NDArray[np.float64]
    vols: NDArray[np.float64]

    @classmethod
    def from_quotes(
        cls,
        quotes: pl.LazyFrame | pl.DataFrame | LazyFrame[VolSurfaceContract],
        *,
        name: str = "vol_surface",
        days_per_year: float = 252.0,
    ) -> VolSurface:
        """Build one rectangular surface from quotes for a single underlying."""
        df = _collect_quotes(quotes)
        if "id_indexador" in df.columns and df["id_indexador"].n_unique() > 1:
            raise ValueError(
                "VolSurface.from_quotes expects a single id_indexador; use VolSurfaceBook"
            )

        tenor_days = np.sort(df["tenor_days"].unique().to_numpy())
        strikes = np.sort(df["strike"].unique().to_numpy()).astype(np.float64)
        grid = (
            df.pivot(on="strike", index="tenor_days", values="implied_vol", sort_columns=True)
            .sort("tenor_days")
            .drop("tenor_days")
        )
        vols = grid.to_numpy().astype(np.float64)
        if np.isnan(vols).any():
            raise ValueError(f"vol surface {name!r} has missing (tenor, strike) cells")

        return cls(
            name=name,
            tenors=tenor_days.astype(np.float64) / days_per_year,
            strikes=strikes,
            vols=vols,
        )

    def implied_vol(self, ttm: ArrayLike, strike: ArrayLike) -> NDArray[np.float64]:
        """Interpolated implied vol at time-to-maturity in years and strike."""
        return bilinear(strike, ttm, self.strikes, self.tenors, self.vols)


@dataclass(frozen=True, slots=True)
class VolSurfaceBook:
    """Collection of one volatility surface per ``id_indexador``."""

    surfaces: dict[int, VolSurface]
    days_per_year: float = 252.0

    @classmethod
    def from_quotes(
        cls,
        quotes: pl.LazyFrame | pl.DataFrame | LazyFrame[VolSurfaceContract],
        *,
        days_per_year: float = 252.0,
    ) -> VolSurfaceBook:
        df = _collect_quotes(quotes)
        surfaces: dict[int, VolSurface] = {}
        for row in df.select("id_indexador").unique().sort("id_indexador").iter_rows(named=True):
            indexer = int(row["id_indexador"])
            surface_df = df.filter(pl.col("id_indexador") == indexer)
            surfaces[indexer] = VolSurface.from_quotes(
                surface_df,
                name=f"vol_surface[{indexer}]",
                days_per_year=days_per_year,
            )
        return cls(surfaces=surfaces, days_per_year=days_per_year)

    def implied_vol(
        self,
        indexer: ArrayLike,
        tenor_days: ArrayLike,
        strike: ArrayLike,
    ) -> NDArray[np.float64]:
        indexer_arr = np.asarray(indexer)
        tenor_arr = np.asarray(tenor_days, dtype=np.float64)
        strike_arr = np.asarray(strike, dtype=np.float64)
        out = np.empty(strike_arr.shape, dtype=np.float64)
        for raw_indexer in np.unique(indexer_arr):
            key = int(raw_indexer)
            surface = self.surfaces.get(key)
            if surface is None:
                known = sorted(self.surfaces)
                raise ValueError(f"unknown id_indexador {key!r} for vol surface; known: {known}")
            mask = indexer_arr == raw_indexer
            ttm = tenor_arr[mask] / self.days_per_year
            out[mask] = surface.implied_vol(ttm, strike_arr[mask])
        return out


@dataclass(frozen=True, slots=True)
class VolSurfaceRequirement:
    table: str
    indexer_col: str
    tenor_col: str
    strike_col: str
    output: str
    days_per_year: float = 252.0

    @property
    def outputs(self) -> dict[str, str]:
        return {"implied_vol": self.output}

    @property
    def left_keys(self) -> tuple[str, ...]:
        return (self.indexer_col, self.tenor_col, self.strike_col)

    @property
    def right_keys(self) -> tuple[str, ...]:
        return ("id_indexador", "tenor_days", "strike")

    def attach(self, lf: pl.LazyFrame, snapshot: MarketSnapshot) -> pl.LazyFrame:
        book = VolSurfaceBook.from_quotes(
            snapshot.source(self.table).data,
            days_per_year=self.days_per_year,
        )
        fields = [self.indexer_col, self.tenor_col, self.strike_col]

        def interpolate(s: pl.Series) -> pl.Series:
            indexer = s.struct.field(self.indexer_col).to_numpy()
            tenor_days = s.struct.field(self.tenor_col).to_numpy()
            strike = s.struct.field(self.strike_col).to_numpy()
            return pl.Series(self.output, book.implied_vol(indexer, tenor_days, strike))

        collisions = sorted({self.output} & set(lf.collect_schema().names()))
        if collisions:
            lf = lf.drop(collisions)
        return lf.with_columns(
            pl.struct(fields).map_batches(interpolate, return_dtype=pl.Float64).alias(self.output)
        )


@dataclass(frozen=True, slots=True)
class VolSurfaceSpec:
    name: str = "vol_surface"

    def implied_vol(
        self,
        *,
        indexer_col: str = "id_indexador",
        tenor_col: str = "payment_days",
        strike_col: str = "strike",
        output: str = "vol",
        days_per_year: float = 252.0,
    ) -> VolSurfaceRequirement:
        return VolSurfaceRequirement(
            table=self.name,
            indexer_col=indexer_col,
            tenor_col=tenor_col,
            strike_col=strike_col,
            output=output,
            days_per_year=days_per_year,
        )


@dataclass(frozen=True, slots=True)
class VolSurfaces:
    data: LazyFrame[VolSurfaceContract]
    name: str = "vol_surface"

    @classmethod
    @pa.check_types(lazy=True)
    def build(
        cls,
        data: LazyFrame[VolSurfaceContract],
        *,
        name: str = "vol_surface",
    ) -> VolSurfaces:
        return cls(data=data, name=name)

    def source(self) -> MarketSource:
        return MarketSource(name=self.name, data=self.data, schema=VolSurfaceContract)

    def spec(self) -> VolSurfaceSpec:
        return VolSurfaceSpec(name=self.name)
