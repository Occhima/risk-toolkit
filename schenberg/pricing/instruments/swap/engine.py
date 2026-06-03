"""Backward-compatible imports for the refactored swap pricing modules."""

from schenberg.pricing.instruments.swap.legs.cdi import cdi_cashflow_graph, cdi_swap_leg_graph
from schenberg.pricing.instruments.swap.legs.ipca import (
    cpi_swap_leg_graph,
    inflation_factor,
    ipca_swap_leg_graph,
    real_coupon_factor,
)
from schenberg.pricing.instruments.swap.legs.ipca import (
    ipca_cashflow_graph as inflation_cashflow_graph,
)
from schenberg.pricing.instruments.swap.pricing import price_swap, price_swaps
from schenberg.pricing.instruments.swap.router import swap_leg_router as swap_router

__all__ = [
    "cdi_cashflow_graph",
    "cdi_swap_leg_graph",
    "cpi_swap_leg_graph",
    "inflation_cashflow_graph",
    "inflation_factor",
    "ipca_swap_leg_graph",
    "price_swap",
    "price_swaps",
    "real_coupon_factor",
    "swap_router",
]
