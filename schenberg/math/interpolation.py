"""Interpolation — agnostic, vectorized, no Polars, no domain.

Two functions, both with flat (clamped) extrapolation so a query off the grid
returns the nearest edge instead of exploding:

* :func:`interp_linear` — 1-D, a thin clamp-extrapolating wrapper over ``np.interp``.
* :func:`bilinear` — 2-D on a rectangular grid; strike-then-tenor for a vol surface.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def interp_linear(x: ArrayLike, xs: ArrayLike, ys: ArrayLike) -> NDArray[np.float64]:
    """Piecewise-linear 1-D interpolation with flat extrapolation.

    ``xs`` must be ascending. ``np.interp`` already clamps outside the range,
    which is exactly the flat extrapolation we want for curves and tenors.
    """
    return np.interp(np.asarray(x, dtype=np.float64), np.asarray(xs), np.asarray(ys))


def bilinear(
    x: ArrayLike,
    y: ArrayLike,
    xs: ArrayLike,
    ys: ArrayLike,
    grid: ArrayLike,
) -> NDArray[np.float64]:
    """Bilinear interpolation on a rectangular grid, clamped at the edges.

    ``grid[i, j]`` is the value at ``(ys[i], xs[j])``; both axes ascending.
    Interpolates along ``x`` (the columns) for every ``y`` row, then along
    ``y``. Fully vectorized over the query points.
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    # 1. collapse the strike axis: per tenor row, interp at the query strikes.
    per_row = np.stack([np.interp(x, xs, grid[i]) for i in range(ys.size)])  # (n_y, n_q)

    # 2. interpolate across tenor rows at each query y (clamped).
    yc = np.clip(y, ys[0], ys[-1])
    lo = np.clip(np.searchsorted(ys, yc, side="right") - 1, 0, ys.size - 2)
    y0, y1 = ys[lo], ys[lo + 1]
    w = np.where(y1 > y0, (yc - y0) / (y1 - y0), 0.0)

    cols = np.arange(x.size)
    low = per_row[lo, cols]
    high = per_row[lo + 1, cols]
    return low + w * (high - low)
