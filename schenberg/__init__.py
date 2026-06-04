"""Schenberg: a lazy, contract-oriented pricing DSL.

Inputs, market reads and formulas are :class:`Term`\\ s inside a
:class:`FormulaGraph`; a :class:`MarketSnapshot` supplies the environment at
compute time; a :class:`Router` is a contract-oriented choice among graphs. On top
of that, a :class:`Structure` composes pure component pricing with exposure and a
:class:`Fold` (monoidal aggregation); a :class:`Shock` is an endomorphism on the
market a :class:`MarketPath` focuses; a :class:`Workflow` runs shape-changing
stages; a :class:`DiagnosticReport` accumulates validation issues.
"""

from schenberg.core.diagnostics import Diagnostic, DiagnosticReport
from schenberg.core.fold import Fold, count_, first_, lit_, sum_
from schenberg.core.graph import FormulaGraph, Term, uses
from schenberg.core.market import MarketDependency, MarketRead, MarketRequirement
from schenberg.core.pipeline import Workflow
from schenberg.core.router import Router
from schenberg.core.structure import Structure
from schenberg.market_data.curves import CurveSpec
from schenberg.market_data.path import MarketPath
from schenberg.market_data.shocks import Shock
from schenberg.market_data.volatility import VolSurfaces, VolSurfaceSpec
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
    "CurveSpec",
    "Diagnostic",
    "DiagnosticReport",
    "Fold",
    "FormulaGraph",
    "MarketDependency",
    "MarketPath",
    "MarketRead",
    "MarketRequirement",
    "Router",
    "Shock",
    "Structure",
    "Term",
    "VolSurfaceSpec",
    "VolSurfaces",
    "Workflow",
    "count_",
    "first_",
    "lit_",
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
