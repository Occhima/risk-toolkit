from __future__ import annotations

import math

import numpy as np
import pytest
from schenberg.math.statistics import norm_cdf, norm_pdf


def test_norm_cdf_known_points() -> None:
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert float(norm_cdf(1.959963985)) == pytest.approx(0.975, abs=1e-6)
    assert float(norm_cdf(-1.959963985)) == pytest.approx(0.025, abs=1e-6)


def test_norm_cdf_symmetry_and_tails() -> None:
    x = np.linspace(-5, 5, 51)
    assert np.allclose(norm_cdf(x) + norm_cdf(-x), 1.0)
    assert norm_cdf(8.0) == pytest.approx(1.0)
    assert norm_cdf(-8.0) == pytest.approx(0.0, abs=1e-12)


def test_norm_pdf_peak_and_integral() -> None:
    assert norm_pdf(0.0) == pytest.approx(1.0 / math.sqrt(2 * math.pi))
    grid = np.linspace(-10, 10, 200_001)
    assert np.trapezoid(norm_pdf(grid), grid) == pytest.approx(1.0, abs=1e-6)


def test_pdf_is_derivative_of_cdf() -> None:
    x = np.linspace(-3, 3, 25)
    h = 1e-6
    fd = (norm_cdf(x + h) - norm_cdf(x - h)) / (2 * h)
    assert np.allclose(fd, norm_pdf(x), atol=1e-7)
