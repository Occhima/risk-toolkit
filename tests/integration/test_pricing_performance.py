"""Pricing throughput on a growing book of swaps.

Not a microbenchmark — a guardrail. It prices books of increasing size end to
end (normalize -> route -> market join -> aggregate -> collect), reports
throughput, and asserts the largest book stays under a generous wall-clock
budget so a perf regression shows up in CI.

Run just this file with timing output:
    uv run pytest tests/integration/test_pricing_performance.py -m performance -s
"""

from __future__ import annotations

import time
from typing import cast

import polars as pl
import pytest
from schenberg.pricing.api import price_swaps

from .realistic_data import make_market, make_swap_legs

# (n_swaps, wall-clock budget in seconds) — budgets are deliberately loose.
SIZES = [(1_000, 3.0), (10_000, 6.0), (50_000, 20.0)]


def _price_and_time(n: int, market) -> tuple[int, float]:
    legs = make_swap_legs(n)
    start = time.perf_counter()
    result = cast(pl.DataFrame, price_swaps(legs, market).collect())
    elapsed = time.perf_counter() - start
    assert result.height == n  # one NPV row per swap
    assert result["npv"].is_finite().all()
    return result.height, elapsed


@pytest.fixture(scope="module")
def market():
    return make_market()


@pytest.mark.performance
@pytest.mark.parametrize("n, budget", SIZES)
def test_swap_pricing_throughput(market, n: int, budget: float) -> None:
    # warm up once so import/JIT/compile costs don't skew the smallest size
    _price_and_time(1, market)

    rows, elapsed = _price_and_time(n, market)
    throughput = rows / elapsed
    print(f"\npriced {rows:>7,} swaps in {elapsed:6.3f}s  ->  {throughput:>10,.0f} swaps/s")

    assert elapsed < budget, f"pricing {n} swaps took {elapsed:.3f}s (budget {budget}s)"


@pytest.mark.performance
def test_scaling_is_better_than_quadratic(market) -> None:
    """10x the rows should cost well under 10x the time once warmed up (the work
    is a single vectorized Polars query, not a Python row loop)."""
    _price_and_time(1, market)
    _, t_small = _price_and_time(1_000, market)
    _, t_large = _price_and_time(10_000, market)
    # generous ceiling: 10x data must not cost more than 8x time
    assert t_large < t_small * 8.0
