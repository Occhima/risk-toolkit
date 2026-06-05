"""The swap as a :class:`Structure`: pure leg pricing + exposure + fold.

This is where the swap's *direction* lives — and the only place it does. The
leg graphs (:mod:`.generic`, :mod:`.legs`) price each leg purely; this structure:

1. routes legs to those pure pricers via ``swap_leg_router`` (the ``pricing`` view);
2. applies exposure ``weighted_pv = pv * leg_weight`` — the pay/receive sign;
3. folds by ``swap_id`` into an NPV plus an ativo/passivo reporting split.

``ativo_pv`` / ``passivo_pv`` are *fold* classifications (filtered sums on
``leg_role``), not pricing formulas.
"""

from __future__ import annotations

import polars as pl

from schenberg.core.columns import cols
from schenberg.core.fold import sum_
from schenberg.core.structure import Structure
from schenberg.domain.schemas import SwapLegInput, SwapOutput

# Import side-effect leg registrations so the router knows every kind.
from schenberg.pricing.instruments.swap import legs as _legs  # noqa: F401
from schenberg.pricing.instruments.swap.router import swap_leg_router

L = cols(SwapLegInput)

swap_structure = (
    Structure("swap", input=SwapLegInput)
    .components(swap_leg_router, view="output")
    .exposure(
        weighted_pv=pl.col("pv") * pl.col("leg_weight"),
    )
    .fold(
        by="swap_id",
        returns=SwapOutput,
        npv=sum_("weighted_pv"),
        ativo_pv=sum_("weighted_pv", where=L.leg_role == "ativo"),
        passivo_pv=sum_("weighted_pv", where=L.leg_role == "passivo"),
    )
)
