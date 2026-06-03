"""Generalized Black-Scholes-Merton: the price and its Greeks, three ways.

This is the pure-computation core — numpy/autograd in, dict-of-arrays out, no
Polars and no domain types (the boundary contract lives in
:class:`schenberg.domain.schemas.option.OptionGreeks`; the lazy bridge and
backend selection live in :mod:`schenberg.risk.greeks`).

``generalized_price`` is the single source of truth: autograd differentiates it,
the finite-difference engine bumps it, and the closed-form engine is reconciled
against it. To stay scipy-free the normal CDF is built from a custom ``erf``
primitive whose vector-Jacobian product is the *exact* Gaussian density, so
reverse-mode AD recovers the true ``N'`` and matches the analytic Greeks to ~1e-10.

The argument order of ``generalized_price`` is the one contract all three engines
share; it is named once in :class:`BsmArg` so no engine carries a magic index.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any

import autograd.numpy as anp
import numpy as np
from autograd import elementwise_grad as egrad
from autograd.extend import defvjp, primitive
from numpy.typing import NDArray

from schenberg.math.statistics import SQRT2
from schenberg.math.statistics import erf as _erf_raw
from schenberg.math.statistics import norm_cdf as _norm_cdf
from schenberg.math.statistics import norm_pdf as _norm_pdf


class BsmArg(IntEnum):
    """Positional argument contract of :func:`generalized_price`.

    The only place the order is written down: ``autograd`` differentiates by
    position and the finite-difference engine bumps by position, so both refer
    to these names instead of bare integers.
    """

    SPOT = 0
    STRIKE = 1
    RATE = 2
    CARRY = 3
    VOL = 4
    TTM = 5
    ETA = 6


GREEK_NAMES = ("delta", "gamma", "vega", "theta", "rho")

Greeks = dict[str, NDArray[np.float64]]

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


def _norm_cdf_ag(x: Any) -> Any:
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
    Vectorized: pass scalars or broadcastable arrays. Argument order is
    :class:`BsmArg`.
    """
    srt = vol * _sqrt(ttm)
    d1 = (_log(spot / strike) + (carry + 0.5 * vol**2) * ttm) / srt
    d2 = d1 - srt
    carry_spot = spot * _exp((carry - rate) * ttm)
    disc_strike = strike * _exp(-rate * ttm)
    return eta * (carry_spot * _norm_cdf_ag(eta * d1) - disc_strike * _norm_cdf_ag(eta * d2))


# --- closed form -------------------------------------------------------------
def greeks_analytic(spot, strike, rate, carry, vol, ttm, eta) -> Greeks:  # noqa: PLR0914
    """Closed-form partials, each holding the others fixed.

    All five have clean analytic forms — the catch is they must be taken with
    respect to the *same* independent variables as the other two engines
    (notably ``rho = dV/dr`` at fixed carry ``b`` and ``theta = -dV/dT``) for
    the three to reconcile.
    """
    spot, strike, rate, carry, vol, ttm, eta = (
        np.asarray(v, dtype=np.float64) for v in (spot, strike, rate, carry, vol, ttm, eta)
    )
    srt = vol * np.sqrt(ttm)
    d1 = (np.log(spot / strike) + (carry + 0.5 * vol**2) * ttm) / srt
    d2 = d1 - srt
    e_br = np.exp((carry - rate) * ttm)
    e_r = np.exp(-rate * ttm)
    n_d1 = _norm_cdf(eta * d1)
    n_d2 = _norm_cdf(eta * d2)
    pdf = _norm_pdf(d1)

    delta = eta * e_br * n_d1
    gamma = e_br * pdf / (spot * srt)
    vega = spot * e_br * pdf * np.sqrt(ttm)
    theta = (
        -(spot * e_br * pdf * vol) / (2.0 * np.sqrt(ttm))
        - eta * (carry - rate) * spot * e_br * n_d1
        - eta * rate * strike * e_r * n_d2
    )
    rho = eta * ttm * (strike * e_r * n_d2 - spot * e_br * n_d1)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# --- finite differences ------------------------------------------------------
# Bump sizes for the finite-difference engine. Spot takes a relative bump (it
# reads more naturally); the rest are small absolute steps. Central differences
# are second-order accurate, so this is the slow-but-obvious reference the other
# two engines are checked against.
_REL_SPOT_BUMP = 1e-4
_ABS_BUMP: dict[BsmArg, float] = {
    BsmArg.VOL: 1e-5,
    BsmArg.RATE: 1e-6,
    BsmArg.TTM: 1e-5,
}


def _price_at(args: list, arg: BsmArg, value) -> NDArray[np.float64]:
    """Price with one argument overridden by ``value``."""
    bumped = list(args)
    bumped[arg] = value
    return generalized_price(*bumped)


def _central(args: list, arg: BsmArg, h) -> NDArray[np.float64]:
    """Central first derivative wrt ``arg`` with step ``h``."""
    return (_price_at(args, arg, args[arg] + h) - _price_at(args, arg, args[arg] - h)) / (2.0 * h)


def greeks_numeric(spot, strike, rate, carry, vol, ttm, eta) -> Greeks:
    args = [np.asarray(v, dtype=np.float64) for v in (spot, strike, rate, carry, vol, ttm, eta)]
    h_spot = _REL_SPOT_BUMP * args[BsmArg.SPOT]

    base = generalized_price(*args)
    up = _price_at(args, BsmArg.SPOT, args[BsmArg.SPOT] + h_spot)
    dn = _price_at(args, BsmArg.SPOT, args[BsmArg.SPOT] - h_spot)

    return {
        "delta": (up - dn) / (2.0 * h_spot),
        "gamma": (up - 2.0 * base + dn) / (h_spot**2),
        "vega": _central(args, BsmArg.VOL, _ABS_BUMP[BsmArg.VOL]),
        "theta": -_central(args, BsmArg.TTM, _ABS_BUMP[BsmArg.TTM]),  # theta = dV/dt = -dV/dT
        "rho": _central(args, BsmArg.RATE, _ABS_BUMP[BsmArg.RATE]),
    }


# --- automatic differentiation -----------------------------------------------
# elementwise_grad differentiates the vectorized price wrt one input at a time;
# gamma is AD applied twice. No hand-derived formulas — the derivatives come
# straight from the model.
# autograd checks ``type(argnum) is int``, so feed it a plain int, not the enum.
_delta = egrad(generalized_price, int(BsmArg.SPOT))
_gamma = egrad(_delta, int(BsmArg.SPOT))
_vega = egrad(generalized_price, int(BsmArg.VOL))
_rho = egrad(generalized_price, int(BsmArg.RATE))
_dV_dT = egrad(generalized_price, int(BsmArg.TTM))


def greeks_autodiff(spot, strike, rate, carry, vol, ttm, eta) -> Greeks:
    args = [np.asarray(v, dtype=np.float64) for v in (spot, strike, rate, carry, vol, ttm, eta)]
    return {
        "delta": _delta(*args),
        "gamma": _gamma(*args),
        "vega": _vega(*args),
        "theta": -_dV_dT(*args),  # theta = dV/dt = -dV/dT
        "rho": _rho(*args),
    }
