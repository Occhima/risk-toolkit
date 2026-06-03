"""Finite-difference Greeks — bump-and-revalue on the shared price model.

Central differences (second-order accurate); gamma is the standard three-point
second difference. Step sizes are small absolute bumps, scaled by spot where a
relative bump reads more naturally. These are the slow-but-obvious reference the
analytic and autograd engines are checked against.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from schenberg.pricing.greeks.model import generalized_price

Greeks = dict[str, NDArray[np.float64]]

# (argument index, absolute step). Spot uses a relative step inside the helpers.
_STEP_VOL = 1e-5
_STEP_RATE = 1e-6
_STEP_TTM = 1e-5


def _bump(args, idx, h):
    up = list(args)
    dn = list(args)
    up[idx] = args[idx] + h
    dn[idx] = args[idx] - h
    return up, dn


def greeks_numeric(spot, strike, rate, carry, vol, ttm, eta) -> Greeks:
    args = [np.asarray(v, dtype=np.float64) for v in (spot, strike, rate, carry, vol, ttm, eta)]
    h_spot = 1e-4 * args[0]

    up, dn = _bump(args, 0, h_spot)
    delta = (generalized_price(*up) - generalized_price(*dn)) / (2.0 * h_spot)
    gamma = (generalized_price(*up) - 2.0 * generalized_price(*args) + generalized_price(*dn)) / (
        h_spot**2
    )

    up, dn = _bump(args, 4, _STEP_VOL)
    vega = (generalized_price(*up) - generalized_price(*dn)) / (2.0 * _STEP_VOL)

    up, dn = _bump(args, 2, _STEP_RATE)
    rho = (generalized_price(*up) - generalized_price(*dn)) / (2.0 * _STEP_RATE)

    up, dn = _bump(args, 5, _STEP_TTM)
    theta = -(generalized_price(*up) - generalized_price(*dn)) / (2.0 * _STEP_TTM)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
