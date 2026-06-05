"""A custom instrument example: the inflation-linked energy forward.

Shows how to extend the toolkit with an instrument the library has never heard
of -- a new graph, an index-convention registry, and a thin pricer -- all on the
same engine the built-ins use.
"""

from .conventions import CONVENTIONS, InflationConvention, add_reference_date
from .graph import inflation_energy_graph
from .pricer import price_inflation_energy

__all__ = [
    "CONVENTIONS",
    "InflationConvention",
    "add_reference_date",
    "inflation_energy_graph",
    "price_inflation_energy",
]
