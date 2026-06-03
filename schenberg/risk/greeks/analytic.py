"""Closed-form generalized BSM Greeks (the partials, holding the others fixed).

All five have clean analytic forms, contrary to the folklore that only delta
does — the catch is they must be taken with respect to the *same* independent
variables as the autograd/numeric engines (notably ``rho = dV/dr`` at fixed
carry ``b``, and ``theta = dV/dt = -dV/dT``) for the three to reconcile.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from schenberg.math.statistics import norm_cdf, norm_pdf

Greeks = dict[str, NDArray[np.float64]]


def greeks_analytic(spot, strike, rate, carry, vol, ttm, eta) -> Greeks:  # noqa: PLR0914
    spot, strike, rate, carry, vol, ttm, eta = (
        np.asarray(v, dtype=np.float64) for v in (spot, strike, rate, carry, vol, ttm, eta)
    )
    srt = vol * np.sqrt(ttm)
    d1 = (np.log(spot / strike) + (carry + 0.5 * vol**2) * ttm) / srt
    d2 = d1 - srt
    e_br = np.exp((carry - rate) * ttm)
    e_r = np.exp(-rate * ttm)
    n_d1 = norm_cdf(eta * d1)
    n_d2 = norm_cdf(eta * d2)
    pdf = norm_pdf(d1)

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
