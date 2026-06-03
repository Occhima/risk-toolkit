"""Implied-volatility surface as market data.

A surface is a rectangular grid of implied vols over ``(tenor, strike)``. Because
looking up a vol is an *interpolation*, not a key join, it cannot ride the
ordinary ``MarketRequirement`` join machinery — so the surface exposes
:meth:`attach`, a vectorized (``map_batches``, never row-wise) step that adds an
``vol`` column to a lazy frame. Tenors are stored in 252-day year fractions so
the query side can pass a time-to-maturity straight through.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray
from pandera.typing.polars import LazyFrame

from schenberg.domain.schemas.market_data import VolSurfaceContract
from schenberg.market_data.sources import MarketSource
from schenberg.math.interpolation import bilinear


@dataclass(frozen=True, slots=True)
class VolSurface:
    """Bilinear implied-vol surface for a single underlying.

    ``vols[i, j]`` is the implied vol at ``(tenors[i], strikes[j])``.
    """

    name: str
    tenors: NDArray[np.float64]  # ascending, in 252-day year fractions
    strikes: NDArray[np.float64]  # ascending
    vols: NDArray[np.float64]  # shape (len(tenors), len(strikes))

    @classmethod
    def from_quotes(
        cls,
        quotes: pl.LazyFrame | pl.DataFrame | LazyFrame[VolSurfaceContract],
        *,
        name: str = "vol_surface",
        days_per_year: float = 252.0,
    ) -> VolSurface:
        """Build a surface by pivoting long-format quotes into a grid.

        Expects one row per ``(tenor_days, strike)`` cell of a rectangular grid;
        a missing cell (a ragged surface) raises rather than guessing.
        """
        df = cast(pl.DataFrame, quotes.lazy().collect())

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
        """Interpolated implied vol at a time-to-maturity (years) and strike."""
        return bilinear(strike, ttm, self.strikes, self.tenors, self.vols)

    def attach(
        self,
        lf: pl.LazyFrame,
        *,
        ttm_col: str = "ttm",
        strike_col: str = "strike",
        output: str = "vol",
    ) -> pl.LazyFrame:
        """Add an implied-vol column by interpolating the surface — stays lazy."""

        def interpolate(s: pl.Series) -> pl.Series:
            ttm = s.struct.field(ttm_col).to_numpy()
            strike = s.struct.field(strike_col).to_numpy()
            return pl.Series(output, self.implied_vol(ttm, strike))

        return lf.with_columns(
            pl.struct([ttm_col, strike_col])
            .map_batches(interpolate, return_dtype=pl.Float64)
            .alias(output)
        )


@dataclass(frozen=True, slots=True)
class VolSurfaceSource:
    """Carries the raw surface quotes inside a :class:`MarketSnapshot`."""

    data: LazyFrame[VolSurfaceContract]
    name: str = "vol_surface"

    def source(self) -> MarketSource:
        return MarketSource(name=self.name, data=self.data, schema=VolSurfaceContract)

    def surface(self) -> VolSurface:
        return VolSurface.from_quotes(self.data, name=self.name)
