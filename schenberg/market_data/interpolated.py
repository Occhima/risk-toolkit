"""Interpolated market data — declare a curve or surface read *between* quotes.

Most market data attaches by an exact key join (see :mod:`schenberg.core.market`).
Some must be *interpolated*: an implied-vol surface, or a zero curve you want to
read between quoted tenors. This module is the single interpolation machine
behind all of them. Quotes arrive as a tidy table — a group key, one or two
ascending axes, and a value — and the requirement interpolates the value at each
row's coordinates: one axis is piecewise-linear, two are bilinear, both clamped
at the edges.

It is declared the same way as any other market data — a ``Spec`` whose method
returns an attachable requirement::

    SURFACE = InterpolatedSpec("vol_surface", axes=("tenor_days", "strike"))
    m = g.market(vol=SURFACE.value("implied_vol", on=("payment_days", "strike")))

    RATES = InterpolatedSpec("rates_curve", axes=("tenor_days",))
    m = g.market(zero_rate=RATES.value("zero_rate", on=("payment_days",)))

Interpolation is scale-invariant along an axis, so axes are interpolated in
their raw quoted units (e.g. tenor in days) — no day-count conversion needed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast, overload

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from schenberg.core.columns import ColumnLike, col_name
from schenberg.core.market import MarketRead
from schenberg.math.interpolation import bilinear, interp_linear

if TYPE_CHECKING:
    from schenberg.market_data.snapshot import MarketSnapshot


@dataclass(frozen=True, slots=True)
class _Grid:
    """One group's values on a 1-D or 2-D ascending axis grid."""

    axes: tuple[NDArray[np.float64], ...]
    values: NDArray[np.float64]

    def at(self, *coords: NDArray[np.float64]) -> NDArray[np.float64]:
        if len(self.axes) == 1:
            return interp_linear(coords[0], self.axes[0], self.values)
        row_axis, col_axis = self.axes
        # values[i, j] sits at (row_axis[i], col_axis[j]); bilinear takes the
        # fast (column) axis first.
        return bilinear(coords[1], coords[0], col_axis, row_axis, self.values)


@dataclass(frozen=True, slots=True)
class InterpolatedBook:
    """One interpolation grid per group key, built from tidy quotes."""

    grids: dict[object, _Grid]

    @classmethod
    def from_quotes(
        cls,
        quotes: pl.LazyFrame | pl.DataFrame,
        *,
        group_col: str,
        axis_cols: tuple[str, ...],
        value_col: str,
    ) -> InterpolatedBook:
        if len(axis_cols) not in (1, 2):
            raise ValueError(f"interpolated market data takes 1 or 2 axes, got {axis_cols!r}")
        df = cast(pl.DataFrame, quotes.collect()) if isinstance(quotes, pl.LazyFrame) else quotes
        grids: dict[object, _Grid] = {}
        for key in df.select(group_col).unique().sort(group_col).to_series().to_list():
            sub = df.filter(pl.col(group_col) == key)
            grids[key] = cls._grid(sub, axis_cols, value_col, key)
        return cls(grids=grids)

    @staticmethod
    def _grid(sub: pl.DataFrame, axis_cols: tuple[str, ...], value_col: str, key: object) -> _Grid:
        axis0 = np.sort(sub[axis_cols[0]].unique().to_numpy()).astype(np.float64)
        if len(axis_cols) == 1:
            ordered = sub.sort(axis_cols[0])[value_col].to_numpy().astype(np.float64)
            return _Grid(axes=(axis0,), values=ordered)
        axis1 = np.sort(sub[axis_cols[1]].unique().to_numpy()).astype(np.float64)
        grid = (
            sub.pivot(on=axis_cols[1], index=axis_cols[0], values=value_col, sort_columns=True)
            .sort(axis_cols[0])
            .drop(axis_cols[0])
            .to_numpy()
            .astype(np.float64)
        )
        if np.isnan(grid).any():
            raise ValueError(f"interpolation grid for group {key!r} has missing cells")
        return _Grid(axes=(axis0, axis1), values=grid)

    def interpolate(self, group: ArrayLike, *coords: ArrayLike) -> NDArray[np.float64]:
        group_arr = np.asarray(group)
        coord_arrs = [np.asarray(c, dtype=np.float64) for c in coords]
        out = np.empty(coord_arrs[0].shape, dtype=np.float64)
        for raw_key in np.unique(group_arr):
            key = raw_key.item()
            grid = self.grids.get(key)
            if grid is None:
                raise ValueError(f"unknown group {key!r}; known: {sorted(self.grids, key=str)}")
            mask = group_arr == raw_key
            out[mask] = grid.at(*(c[mask] for c in coord_arrs))
        return out


@dataclass(frozen=True, slots=True)
class InterpolatedRequirement:
    """Attachable market data that interpolates a value at each row's coordinates."""

    table: str
    group: tuple[str, str]  # (left/trade column, right/quote column)
    axes: tuple[tuple[str, str], ...]  # per axis: (left/trade column, right/quote column)
    value_col: str
    output: str

    @property
    def outputs(self) -> dict[str, str]:
        return {self.value_col: self.output}

    def with_output(self, output: str) -> InterpolatedRequirement:
        return replace(self, output=output)

    @property
    def left_keys(self) -> tuple[str, ...]:
        return (self.group[0], *(a[0] for a in self.axes))

    @property
    def right_keys(self) -> tuple[str, ...]:
        return (self.group[1], *(a[1] for a in self.axes))

    def attach(self, lf: pl.LazyFrame, snapshot: MarketSnapshot) -> pl.LazyFrame:
        book = InterpolatedBook.from_quotes(
            snapshot.source(self.table).data,
            group_col=self.group[1],
            axis_cols=tuple(a[1] for a in self.axes),
            value_col=self.value_col,
        )
        group_left = self.group[0]
        axis_left = [a[0] for a in self.axes]
        fields = [group_left, *axis_left]

        def interpolate(s: pl.Series) -> pl.Series:
            group = s.struct.field(group_left).to_numpy()
            coords = [s.struct.field(a).to_numpy() for a in axis_left]
            return pl.Series(self.output, book.interpolate(group, *coords))

        collisions = sorted({self.output} & set(lf.collect_schema().names()))
        if collisions:
            lf = lf.drop(collisions)
        return lf.with_columns(
            pl.struct(fields).map_batches(interpolate, return_dtype=pl.Float64).alias(self.output)
        )


@dataclass(frozen=True, slots=True)
class InterpolatedSpec:
    """Declare an interpolated market source. ``axes`` names the quote-side axis
    columns (ascending); ``group`` is the quote-side group key."""

    name: str
    axes: tuple[str, ...]
    group: str = "id_indexador"

    @overload
    def value(
        self,
        value_col: str,
        *,
        output: None = ...,
        on: tuple[ColumnLike, ...] | None = ...,
        group_col: ColumnLike | None = ...,
    ) -> MarketRead: ...

    @overload
    def value(
        self,
        value_col: str,
        *,
        output: str,
        on: tuple[ColumnLike, ...] | None = ...,
        group_col: ColumnLike | None = ...,
    ) -> InterpolatedRequirement: ...

    def value(
        self,
        value_col: str,
        *,
        output: str | None = None,
        on: tuple[ColumnLike, ...] | None = None,
        group_col: ColumnLike | None = None,
    ) -> InterpolatedRequirement | MarketRead:
        """Build the attachable read.

        ``on`` names the trade-side axis columns (defaults to the quote axes);
        ``group_col`` the trade-side group key. With ``output`` omitted it returns
        a delayed :class:`MarketRead` named later by ``g.market``; with ``output``
        given it returns the concrete :class:`InterpolatedRequirement`.
        """
        raw_axes = on if on is not None else self.axes
        if len(raw_axes) != len(self.axes):
            raise ValueError(f"expected {len(self.axes)} axis column(s) in `on`, got {raw_axes!r}")
        trade_axes = tuple(col_name(a) for a in raw_axes)
        group_left = col_name(group_col) if group_col is not None else self.group

        def build(out: str) -> InterpolatedRequirement:
            return InterpolatedRequirement(
                table=self.name,
                group=(group_left, self.group),
                axes=tuple(zip(trade_axes, self.axes, strict=True)),
                value_col=value_col,
                output=out,
            )

        if output is None:
            return MarketRead(build=build)
        return build(output)
