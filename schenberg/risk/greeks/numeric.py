"""Finite-difference Greeks — bump-and-revalue on the shared price model.

Central differences (second-order accurate); gamma is the standard three-point
second difference. Step sizes are small absolute bumps, scaled by spot where a
relative bump reads more naturally. These are the slow-but-obvious reference the
analytic and autograd engines are checked against.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from schenberg.risk.greeks.model import generalized_price

Greeks = dict[str, NDArray[np.float64]]

# Absolute bump sizes by argument; spot uses a relative bump (see below).
_STEP_VOL = 1e-5
_STEP_RATE = 1e-6
_STEP_TTM = 1e-5

# Argument positions in generalized_price(spot, strike, rate, carry, vol, ttm, eta).
_SPOT, _RATE, _VOL, _TTM = 0, 2, 4, 5


def _price_at(args: list, idx: int, value) -> NDArray[np.float64]:
    """Price with argument ``idx`` overridden by ``value``."""
    bumped = list(args)
    bumped[idx] = value
    return generalized_price(*bumped)


def _central(args: list, idx: int, h) -> NDArray[np.float64]:
    """Central first derivative wrt argument ``idx`` with step ``h``."""
    return (_price_at(args, idx, args[idx] + h) - _price_at(args, idx, args[idx] - h)) / (2.0 * h)


def greeks_numeric(spot, strike, rate, carry, vol, ttm, eta) -> Greeks:
    args = [np.asarray(v, dtype=np.float64) for v in (spot, strike, rate, carry, vol, ttm, eta)]
    h_spot = 1e-4 * args[_SPOT]

    base = generalized_price(*args)
    up = _price_at(args, _SPOT, args[_SPOT] + h_spot)
    dn = _price_at(args, _SPOT, args[_SPOT] - h_spot)

    return {
        "delta": (up - dn) / (2.0 * h_spot),
        "gamma": (up - 2.0 * base + dn) / (h_spot**2),
        "vega": _central(args, _VOL, _STEP_VOL),
        "theta": -_central(args, _TTM, _STEP_TTM),  # theta = dV/dt = -dV/dT
        "rho": _central(args, _RATE, _STEP_RATE),
    }
