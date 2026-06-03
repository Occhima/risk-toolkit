"""The canonical generalized BSM price, written once in ``autograd.numpy``.

This single scalar/vectorized function is the shared source of truth for all
three Greek engines: autograd differentiates it, the finite-difference engine
bumps it, and the closed-form engine is checked against it.

To stay scipy-free (and therefore Windows-clean) the normal CDF is built from a
custom ``erf`` primitive whose vector-Jacobian product is the *exact* Gaussian
density — so reverse-mode AD recovers the true ``N'`` and matches the analytic
Greeks to ~1e-10.
"""

from __future__ import annotations

import math
from typing import Any

import autograd.numpy as anp
from autograd.extend import defvjp, primitive

from schenberg.math.statistics import SQRT2
from schenberg.math.statistics import erf as _erf_raw

# autograd.numpy mirrors numpy at runtime but ships no type stubs; bind the few
# ufuncs we need once (as Any) so the math below type-checks cleanly.
_exp: Any = anp.exp  # ty: ignore[unresolved-attribute]
_sqrt: Any = anp.sqrt  # ty: ignore[unresolved-attribute]
_log: Any = anp.log  # ty: ignore[unresolved-attribute]

_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)

# Differentiable erf: forward uses the agnostic (math.erf-backed) implementation;
# backward injects erf'(x) = 2/sqrt(pi) * e^{-x^2}, written in autograd.numpy so
# it is itself differentiable (gamma is AD applied twice).
erf = primitive(_erf_raw)
defvjp(erf, lambda ans, x: lambda g: g * _TWO_OVER_SQRT_PI * _exp(-x * x))


def norm_cdf(x: Any) -> Any:
    """Autograd-traceable standard normal CDF."""
    return 0.5 * (1.0 + erf(x / SQRT2))


def generalized_price(
    spot: Any,
    strike: Any,
    rate: Any,
    carry: Any,
    vol: Any,
    ttm: Any,
    eta: Any,
) -> Any:
    """Generalized Black-Scholes-Merton price.

    ``eta`` is +1 for a call, -1 for a put. ``carry`` is the cost of carry ``b``,
    treated as an independent variable (so ``rho`` is ``dV/dr`` at fixed ``b``).
    Vectorized: pass scalars or broadcastable arrays.
    """
    srt = vol * _sqrt(ttm)
    d1 = (_log(spot / strike) + (carry + 0.5 * vol**2) * ttm) / srt
    d2 = d1 - srt
    carry_spot = spot * _exp((carry - rate) * ttm)
    disc_strike = strike * _exp(-rate * ttm)
    return eta * (carry_spot * norm_cdf(eta * d1) - disc_strike * norm_cdf(eta * d2))
