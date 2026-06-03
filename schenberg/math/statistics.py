"""Standard-normal statistics — agnostic, scalar/vectorized, no Polars, no domain.

Kept deliberately tiny: the error function (via :func:`math.erf`, machine
accurate) is the only primitive; the normal CDF and PDF fall out of it. These
power the closed-form option Greeks and are the *same* functions the autograd
layer differentiates (it wraps :func:`erf` with the exact Gaussian derivative),
so analytic and automatic Greeks agree to ~1e-10.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

_erf = np.vectorize(math.erf, otypes=[np.float64])


def erf(x: ArrayLike) -> NDArray[np.float64]:
    """Gauss error function, elementwise and machine-accurate."""
    return _erf(np.asarray(x, dtype=np.float64))


def norm_pdf(x: ArrayLike) -> NDArray[np.float64]:
    """Standard-normal probability density."""
    x = np.asarray(x, dtype=np.float64)
    return INV_SQRT_2PI * np.exp(-0.5 * x * x)


def norm_cdf(x: ArrayLike) -> NDArray[np.float64]:
    """Standard-normal cumulative distribution, N(x) = ½(1 + erf(x/√2))."""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + erf(x / SQRT2))
