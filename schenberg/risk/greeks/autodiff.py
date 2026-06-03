"""Autograd Greeks — reverse-mode AD of the shared price model.

``elementwise_grad`` differentiates the vectorized price with respect to one
input at a time; gamma is AD applied twice. This is the engine the user wanted:
no hand-derived formulas, the derivatives come straight from the model.
"""

from __future__ import annotations

import numpy as np
from autograd import elementwise_grad as egrad
from numpy.typing import NDArray

from schenberg.risk.greeks.model import generalized_price

Greeks = dict[str, NDArray[np.float64]]

# argument order: spot=0, strike=1, rate=2, carry=3, vol=4, ttm=5, eta=6
_delta = egrad(generalized_price, 0)
_gamma = egrad(_delta, 0)
_vega = egrad(generalized_price, 4)
_rho = egrad(generalized_price, 2)
_dV_dT = egrad(generalized_price, 5)


def greeks_autodiff(spot, strike, rate, carry, vol, ttm, eta) -> Greeks:
    args = [np.asarray(v, dtype=np.float64) for v in (spot, strike, rate, carry, vol, ttm, eta)]
    return {
        "delta": _delta(*args),
        "gamma": _gamma(*args),
        "vega": _vega(*args),
        "theta": -_dV_dT(*args),  # theta = dV/dt = -dV/dT
        "rho": _rho(*args),
    }
