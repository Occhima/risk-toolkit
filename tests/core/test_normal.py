from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from schenberg import compile_numeric, compile_polars, normal_cdf, normal_pdf, to_latex, var


def test_normal_numeric_values() -> None:
    np_backend = np
    assert compile_numeric(normal_cdf(0.0))({}, np_backend) == pytest.approx(0.5, abs=1e-7)
    assert compile_numeric(normal_pdf(0.0))({}, np_backend) == pytest.approx(
        1 / math.sqrt(2 * math.pi)
    )
    assert compile_numeric(normal_cdf(1.0))({}, np_backend) == pytest.approx(0.841344746, abs=1e-6)
    assert compile_numeric(normal_cdf(-1.0))({}, np_backend) == pytest.approx(0.158655254, abs=1e-6)


def test_normal_polars_and_latex() -> None:
    lf = (
        pl.DataFrame({"d1": [0.0, 1.0, -1.0]})
        .lazy()
        .select(
            compile_polars(normal_cdf(var("d1"))).alias("cdf"),
            compile_polars(normal_pdf(var("d1"))).alias("pdf"),
        )
    )
    assert isinstance(lf, pl.LazyFrame)
    out = lf.collect()
    assert out["cdf"].to_list() == pytest.approx([0.5, 0.841344746, 0.158655254], abs=1e-6)
    assert "\\Phi" in to_latex(normal_cdf(var("d1")))
    assert "\\phi" in to_latex(normal_pdf(var("d1")))


def test_expr_has_no_map_elements() -> None:
    assert ".map_elements" not in Path("schenberg/core/expr.py").read_text()
