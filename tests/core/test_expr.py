from __future__ import annotations

import importlib.util
import math

import numpy as np
import polars as pl
import pytest
from schenberg.core.expr import (
    Expr,
    compile_numeric,
    compile_polars,
    exp,
    grad,
    lit,
    to_latex,
    var,
    where,
)

# forward value: (F - K) * exp(-r * T)
F, K, r, T = var("forward_price"), var("strike"), var("risk_free"), var("year_fraction")
FORWARD_VALUE = (F - K) * exp(-r * T)


def test_compile_polars_matches_hand_written() -> None:
    df = pl.DataFrame(
        {
            "forward_price": [5.0, 6.0],
            "strike": [4.0, 5.0],
            "risk_free": [0.10, 0.12],
            "year_fraction": [1.0, 2.0],
        }
    )
    got = df.select(value=compile_polars(FORWARD_VALUE))["value"].to_list()
    want = [
        (5.0 - 4.0) * math.exp(-0.10 * 1.0),
        (6.0 - 5.0) * math.exp(-0.12 * 2.0),
    ]
    assert got == pytest.approx(want)


def test_compile_numeric_matches_polars() -> None:
    env = {
        "forward_price": np.array([5.0, 6.0]),
        "strike": np.array([4.0, 5.0]),
        "risk_free": np.array([0.10, 0.12]),
        "year_fraction": np.array([1.0, 2.0]),
    }
    numeric = compile_numeric(FORWARD_VALUE)(env, np)
    polars = (
        pl.DataFrame({k: v.tolist() for k, v in env.items()})
        .select(v=compile_polars(FORWARD_VALUE))["v"]
        .to_numpy()
    )
    assert numeric == pytest.approx(polars)


def test_where_branches_in_both_backends() -> None:
    payoff = where(F > K, F - K, lit(0.0))
    df = pl.DataFrame({"forward_price": [5.0, 3.0], "strike": [4.0, 4.0]})
    got = df.select(v=compile_polars(payoff))["v"].to_list()
    assert got == [1.0, 0.0]

    env = {"forward_price": np.array([5.0, 3.0]), "strike": np.array([4.0, 4.0])}
    assert compile_numeric(payoff)(env, np).tolist() == [1.0, 0.0]


def test_to_latex_is_derived_from_formula() -> None:
    latex = to_latex(FORWARD_VALUE)
    assert r"\mathrm{forward\_price}" in latex
    assert "e^{" in latex


def test_not_sympy_no_simplification() -> None:
    # x - x stays a subtraction node; the IR never simplifies.
    e = var("x") - var("x")
    assert isinstance(e, Expr)
    assert e.op == "sub"


_HAS_JAX = importlib.util.find_spec("jax") is not None
requires_jax = pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")


@requires_jax
def test_grad_matches_analytic_derivative() -> None:
    # d/dF [(F - K) * exp(-rT)] = exp(-rT)
    df = grad(FORWARD_VALUE, "forward_price")
    env = {
        "forward_price": np.array([5.0, 6.0]),
        "strike": np.array([4.0, 5.0]),
        "risk_free": np.array([0.10, 0.12]),
        "year_fraction": np.array([1.0, 2.0]),
    }
    got = np.asarray(df(env))
    want = np.exp(-env["risk_free"] * env["year_fraction"])
    assert got == pytest.approx(want)


@requires_jax
def test_grad_matches_finite_difference() -> None:
    df = grad(FORWARD_VALUE, "risk_free")
    env = {
        "forward_price": np.array([5.0]),
        "strike": np.array([4.0]),
        "risk_free": np.array([0.10]),
        "year_fraction": np.array([2.0]),
    }
    h = 1e-6
    f = compile_numeric(FORWARD_VALUE)
    up = f({**env, "risk_free": env["risk_free"] + h}, np)
    dn = f({**env, "risk_free": env["risk_free"] - h}, np)
    fd = (up - dn) / (2 * h)
    assert np.asarray(df(env)) == pytest.approx(fd, rel=1e-5)
