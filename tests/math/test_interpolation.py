from __future__ import annotations

import numpy as np
import pytest
from schenberg.math.interpolation import bilinear, interp_linear


def test_interp_linear_nodes_midpoint_and_flat_extrapolation() -> None:
    xs = [0.0, 1.0, 2.0]
    ys = [10.0, 20.0, 40.0]
    assert interp_linear(1.0, xs, ys) == pytest.approx(20.0)
    assert interp_linear(0.5, xs, ys) == pytest.approx(15.0)
    assert interp_linear(1.5, xs, ys) == pytest.approx(30.0)
    # flat extrapolation clamps to the edges
    assert interp_linear(-5.0, xs, ys) == pytest.approx(10.0)
    assert interp_linear(99.0, xs, ys) == pytest.approx(40.0)


def _grid():
    xs = np.array([10.0, 20.0])  # strikes
    ys = np.array([1.0, 2.0])  # tenors
    z = np.array([[0.10, 0.20], [0.30, 0.40]])  # z[tenor, strike]
    return xs, ys, z


def test_bilinear_hits_grid_nodes() -> None:
    xs, ys, z = _grid()
    assert bilinear(10.0, 1.0, xs, ys, z)[0] == pytest.approx(0.10)
    assert bilinear(20.0, 2.0, xs, ys, z)[0] == pytest.approx(0.40)


def test_bilinear_center_is_corner_average() -> None:
    xs, ys, z = _grid()
    assert bilinear(15.0, 1.5, xs, ys, z)[0] == pytest.approx(0.25)


def test_bilinear_axis_aligned_interpolation() -> None:
    xs, ys, z = _grid()
    assert bilinear(15.0, 1.0, xs, ys, z)[0] == pytest.approx(0.15)  # along strike only
    assert bilinear(10.0, 1.5, xs, ys, z)[0] == pytest.approx(0.20)  # along tenor only


def test_bilinear_clamps_outside_the_grid() -> None:
    xs, ys, z = _grid()
    assert bilinear(0.0, 0.0, xs, ys, z)[0] == pytest.approx(0.10)
    assert bilinear(100.0, 100.0, xs, ys, z)[0] == pytest.approx(0.40)


def test_bilinear_is_vectorized() -> None:
    xs, ys, z = _grid()
    out = bilinear([10.0, 20.0, 15.0], [1.0, 2.0, 1.5], xs, ys, z)
    assert np.allclose(out, [0.10, 0.40, 0.25])
