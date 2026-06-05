"""Schenberg: a lazy, contract-oriented pricing DSL.

Inputs, market reads and formulas are :class:`Term`\\ s inside a
:class:`FormulaGraph`; a :class:`MarketSnapshot` supplies the environment at
compute time; a :class:`Router` is a contract-oriented choice among graphs. On top
of that, a :class:`Structure` composes pure component pricing with exposure and a
:class:`Fold` (monoidal aggregation); a :class:`Shock` is an endomorphism on the
market a :class:`MarketPath` focuses; a :class:`Workflow` runs shape-changing
stages; a :class:`DiagnosticReport` accumulates validation issues.
"""

from schenberg.contracts import DataFrameModel, price_function
from schenberg.core.diagnostics import Diagnostic, DiagnosticReport
from schenberg.core.fold import Fold, count_, first_, lit_, sum_
from schenberg.core.graph import FormulaGraph, PricingGraph, Term, uses
from schenberg.core.market import MarketDependency, MarketRequirement
from schenberg.core.pipeline import Workflow
from schenberg.core.router import Router
from schenberg.core.structure import Structure
from schenberg.market_data.path import MarketPath
from schenberg.market_data.requirements import MarketRequirements, contract, requires
from schenberg.market_data.shocks import Shock
from schenberg.pricing.api import (
    price_energy_forward,
    price_forwards,
    price_options,
    price_options_with_greeks,
    price_swap,
    price_swaps,
)

__all__ = [
    # the pricing language
    "DataFrameModel",
    "Diagnostic",
    "DiagnosticReport",
    "Fold",
    "FormulaGraph",
    "MarketDependency",
    "MarketPath",
    "MarketRequirement",
    "MarketRequirements",
    "PricingGraph",
    "Router",
    "Shock",
    "Structure",
    "Term",
    "Workflow",
    "contract",
    "count_",
    "first_",
    "lit_",
    "price_function",
    "requires",
    "sum_",
    "uses",
    # public pricing facades
    "price_energy_forward",
    "price_forwards",
    "price_options",
    "price_options_with_greeks",
    "price_swap",
    "price_swaps",
]
