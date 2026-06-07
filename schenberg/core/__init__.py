"""The Schenberg engine: a symbolic expression IR, AST formula graphs, routing,
folds, market-join dependencies, and diagnostics."""

from schenberg.core.diagnostics import Diagnostic, DiagnosticReport
from schenberg.core.expr import (
    Expr,
    abs_,
    compile_numeric,
    compile_polars,
    exp,
    grad,
    lit,
    log,
    sqrt,
    to_latex,
    var,
    where,
)
from schenberg.core.fold import Agg, Fold, count_, first_, lit_, strict_sum_, sum_
from schenberg.core.graph import Formula, FormulaGraph, GraphInfo, TermMeta
from schenberg.core.market import MarketDependency, MarketRequirement
from schenberg.core.router import Router

__all__ = [
    "Agg",
    "Diagnostic",
    "DiagnosticReport",
    "Expr",
    "Fold",
    "Formula",
    "FormulaGraph",
    "GraphInfo",
    "MarketDependency",
    "MarketRequirement",
    "Router",
    "TermMeta",
    "abs_",
    "compile_numeric",
    "compile_polars",
    "count_",
    "exp",
    "first_",
    "grad",
    "lit",
    "lit_",
    "log",
    "sqrt",
    "sum_",
    "strict_sum_",
    "to_latex",
    "var",
    "where",
]
