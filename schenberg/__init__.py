"""Schenberg: a lazy, contract-oriented pricing DSL.

A formula is an :class:`Expr` tree inside a :class:`FormulaGraph`; it is *pure* —
market data is resolved into its input frame beforehand by :func:`bind` over
declared market :class:`MarketRole`\\ s, never joined inside the graph. The same
expression compiles to lazy Polars now (:func:`compile_polars`), to an analytic
derivative via JAX (:func:`grad`), and to LaTeX (:func:`to_latex`). A
:class:`MarketSnapshot` is the resolved market environment; a :class:`Router` is a
contract-oriented choice among pure computations; a :class:`Fold` handles explicit
aggregation; a :class:`Shock` is an endomorphism on the market a :class:`MarketPath`
focuses; a :class:`PositionView` lifts pure instrument values onto positions.
"""

from schenberg.contracts import SchenbergDataFrameModel, price_function
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
    normal_cdf,
    normal_pdf,
    sqrt,
    to_latex,
    var,
    where,
)
from schenberg.core.fold import Fold, count_, first_, lit_, strict_sum_, sum_
from schenberg.core.graph import Formula, FormulaGraph
from schenberg.core.market import MarketDependency, MarketRequirement
from schenberg.core.router import Router
from schenberg.market_data.path import MarketPath
from schenberg.market_data.roles import (
    Fixing,
    MarketRole,
    With,
    bind,
    market_role,
    roles_of,
)
from schenberg.market_data.semantic import CURVES, FIXINGS, VOLS
from schenberg.market_data.shocks import Shock
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.position import PositionView
from schenberg.risk import Scenario, ScenarioSet, reprice_under
from schenberg.structure import (
    StructureLeg,
    StructureLegValue,
    StructureValue,
    structure_stage,
    structure_value,
    structure_value_fold,
)

__all__ = [
    # the pricing language
    "SchenbergDataFrameModel",
    "price_function",
    # the expression IR + backends
    "Expr",
    "var",
    "lit",
    "exp",
    "log",
    "normal_cdf",
    "normal_pdf",
    "sqrt",
    "abs_",
    "where",
    "compile_polars",
    "compile_numeric",
    "grad",
    "to_latex",
    # the formula graph
    "Formula",
    "FormulaGraph",
    # market binding
    "MarketRole",
    "Fixing",
    "With",
    "bind",
    "market_role",
    "roles_of",
    "MarketSnapshot",
    "MarketSource",
    "MarketDependency",
    "MarketRequirement",
    "MarketPath",
    "Shock",
    "CURVES",
    "FIXINGS",
    "VOLS",
    # composition / aggregation / position
    "Router",
    "Fold",
    "PositionView",
    "count_",
    "first_",
    "lit_",
    "sum_",
    "strict_sum_",
    "Scenario",
    "ScenarioSet",
    "reprice_under",
    "StructureLeg",
    "StructureLegValue",
    "StructureValue",
    "structure_stage",
    "structure_value",
    "structure_value_fold",
    # diagnostics
    "Diagnostic",
    "DiagnosticReport",
]
