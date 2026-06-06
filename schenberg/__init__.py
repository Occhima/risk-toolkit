"""Schenberg: a lazy, contract-oriented pricing DSL.

Inputs, market reads and formulas are :class:`Term`\\ s inside a
:class:`FormulaGraph`; a :class:`MarketSnapshot` supplies the environment at
compute time; a :class:`Router` is a contract-oriented choice among graphs. A
:class:`Fold` handles explicit aggregation, a :class:`Shock` is an endomorphism on
the market a :class:`MarketPath` focuses, and a :class:`DiagnosticReport`
accumulates validation issues.
"""

from schenberg.contracts import SchenbergDataFrameModel, price_function
from schenberg.core.diagnostics import Diagnostic, DiagnosticReport
from schenberg.core.fold import Fold, count_, first_, lit_, sum_
from schenberg.core.graph import Formula, FormulaGraph, Term, uses
from schenberg.core.market import MarketDependency, MarketRequirement
from schenberg.core.router import Router
from schenberg.market_data.path import MarketPath
from schenberg.market_data.requirements import MarketRequirements, contract, requires
from schenberg.market_data.shocks import Shock

__all__ = [
    # the pricing language
    "SchenbergDataFrameModel",
    "Diagnostic",
    "DiagnosticReport",
    "Fold",
    "Formula",
    "FormulaGraph",
    "MarketDependency",
    "MarketPath",
    "MarketRequirement",
    "MarketRequirements",
    "Router",
    "Shock",
    "Term",
    "contract",
    "count_",
    "first_",
    "lit_",
    "price_function",
    "requires",
    "sum_",
    "uses",
]
