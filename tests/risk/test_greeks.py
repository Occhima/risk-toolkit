from __future__ import annotations

import numpy as np
import pytest
from schenberg.domain.schemas.option import OptionGreeks
from schenberg.math.black_scholes import GREEK_NAMES, greeks_analytic
from schenberg.risk.greeks import GreeksBackend, GreeksEngine


def compute_greeks(*, backend, **kwargs):
    return GreeksEngine(GreeksBackend(backend)).compute(**kwargs)


# A spread of moneyness, carries and maturities — calls and puts.
SPOT = np.array([100.0, 100.0, 80.0, 120.0, 100.0, 100.0])
STRIKE = np.array([100.0, 100.0, 100.0, 100.0, 90.0, 110.0])
RATE = np.array([0.10, 0.05, 0.10, 0.02, 0.10, 0.07])
CARRY = np.array([0.10, 0.02, 0.07, 0.00, 0.10, 0.03])  # b: BS / FX / Merton / Black-76 / ...
VOL = np.array([0.20, 0.35, 0.25, 0.15, 0.40, 0.18])
TTM = np.array([1.0, 0.5, 2.0, 0.25, 1.5, 0.75])
CALL = np.ones(6)
PUT = -np.ones(6)


@pytest.mark.parametrize("eta", [CALL, PUT], ids=["call", "put"])
def test_three_methods_reconcile(eta) -> None:
    args = dict(spot=SPOT, strike=STRIKE, rate=RATE, carry=CARRY, vol=VOL, ttm=TTM, eta=eta)
    analytic = compute_greeks(backend="CLOSED_FORM", **args)
    numeric = compute_greeks(backend="NUMERIC", **args)
    autodiff = compute_greeks(backend="AUTODIFF", **args)

    for name in GREEK_NAMES:
        # closed-form and AD share the exact Gaussian -> agree to ~1e-9
        assert np.allclose(analytic[name], autodiff[name], rtol=1e-7, atol=1e-7), name
        # finite differences are second-order accurate -> looser tolerance
        assert np.allclose(analytic[name], numeric[name], rtol=1e-4, atol=1e-4), name


def test_greek_signs_are_sane() -> None:
    call = compute_greeks(
        backend="CLOSED_FORM",
        spot=SPOT,
        strike=STRIKE,
        rate=RATE,
        carry=CARRY,
        vol=VOL,
        ttm=TTM,
        eta=CALL,
    )
    put = compute_greeks(
        backend="CLOSED_FORM",
        spot=SPOT,
        strike=STRIKE,
        rate=RATE,
        carry=CARRY,
        vol=VOL,
        ttm=TTM,
        eta=PUT,
    )
    assert np.all(call["delta"] > 0) and np.all(put["delta"] < 0)
    assert np.all(call["gamma"] > 0) and np.all(put["gamma"] > 0)  # gamma kind-independent
    assert np.all(call["vega"] > 0) and np.all(put["vega"] > 0)  # vega kind-independent
    assert np.allclose(call["gamma"], put["gamma"])
    assert np.allclose(call["vega"], put["vega"])


def test_delta_difference_equals_carry_discount_factor() -> None:
    # Generalized BSM: delta_call - delta_put = e^{(b-r)T}.
    g = greeks_analytic(SPOT, STRIKE, RATE, CARRY, VOL, TTM, CALL)
    p = greeks_analytic(SPOT, STRIKE, RATE, CARRY, VOL, TTM, PUT)
    assert np.allclose(g["delta"] - p["delta"], np.exp((CARRY - RATE) * TTM))


def test_scalar_inputs_work() -> None:
    g = compute_greeks(
        backend="AUTODIFF", spot=100.0, strike=100.0, rate=0.1, carry=0.1, vol=0.2, ttm=1.0, eta=1.0
    )
    assert float(g["delta"]) == pytest.approx(0.7257468822, abs=1e-7)


def test_greek_names_match_the_contract() -> None:
    # The computation layer's names and the OptionGreeks boundary contract must
    # stay in lockstep — both backends key their output off this.
    assert tuple(OptionGreeks.to_schema().columns.keys()) == GREEK_NAMES
