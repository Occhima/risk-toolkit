"""Black-Scholes vanilla option pricing as an ExprGraph.

Demonstrates that non-trivial closed forms fit the same engine: the only enabler
is a pure-Polars normal CDF (no map_elements). Call vs put is one Router over
option_kind, with compose() used to clone the shared graph into two output views.
"""
from __future__ import annotations

import math

import polars as pl

from .graph import ExprGraph, Router


def norm_cdf(x: pl.Expr) -> pl.Expr:
    """Standard normal CDF, Abramowitz & Stegun 26.2.17 — vectorized, pure
    Polars, no map_elements (~7.5e-8 abs error). N(x) = 1 - N(-x)."""
    ax = x.abs()
    t = 1.0 / (1.0 + 0.2316419 * ax)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
                + t * (-1.821255978 + t * 1.330274429))))
    pdf = (-(ax * ax) / 2.0).exp() / math.sqrt(2.0 * math.pi)
    cdf_pos = 1.0 - pdf * poly
    return pl.when(x >= 0).then(cdf_pos).otherwise(1.0 - cdf_pos)


bs_vanilla_graph = ExprGraph("bs_vanilla")


@bs_vanilla_graph.node(tags=("bs",))
def d1(spot, strike, rate, vol, ttm):
    return ((spot / strike).log() + (rate + 0.5 * vol ** 2) * ttm) / (vol * ttm.sqrt())


@bs_vanilla_graph.node(tags=("bs",))
def d2(d1, vol, ttm):
    return d1 - vol * ttm.sqrt()


@bs_vanilla_graph.node(tags=("bs",))
def disc_strike(strike, rate, ttm):          # K * e^(-rT)
    return strike * (-rate * ttm).exp()


@bs_vanilla_graph.node(tags=("bs", "call"))
def call_price(spot, d1, d2, disc_strike):
    return spot * norm_cdf(d1) - disc_strike * norm_cdf(d2)


@bs_vanilla_graph.node(tags=("bs", "put"))
def put_price(spot, d1, d2, disc_strike):
    return disc_strike * norm_cdf(-d2) - spot * norm_cdf(-d1)


# compose() with a single graph = a fresh clone you can configure differently.
bs_call_graph = ExprGraph.compose("bs_call", bs_vanilla_graph).with_outputs(
    "pricing", d1="d1", d2="d2", price="call_price")
bs_put_graph = ExprGraph.compose("bs_put", bs_vanilla_graph).with_outputs(
    "pricing", d1="d1", d2="d2", price="put_price")

bs_router = Router("option_kind", {"CALL": bs_call_graph, "PUT": bs_put_graph})
