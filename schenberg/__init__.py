"""Schenberg: a lazy, contract-oriented pricing DSL.

Inputs, market reads and formulas are :class:`Term`\\ s inside a
:class:`FormulaGraph`; a :class:`MarketSnapshot` supplies the environment at
compute time; a :class:`Router` is a contract-oriented choice among graphs; a
:class:`Workflow` runs shape-changing dataframe stages.
"""

from schenberg.core.graph import FormulaGraph, Term, uses
from schenberg.core.market import MarketDependency, MarketRead, MarketRequirement
from schenberg.core.pipeline import Workflow
from schenberg.core.router import Router
from schenberg.market_data.curves import CurveSpec
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
    "FormulaGraph",
    "MarketDependency",
    "MarketRead",
    "MarketRequirement",
    "Router",
    "Term",
    "VolSurfaceSpec",
    "VolSurfaces",
    "Workflow",
    "uses",
    # public pricing facades
    "price_energy_forward",
    "price_forwards",
    "price_options",
    "price_options_with_greeks",
    "price_swap",
    "price_swaps",
]
