"""Builders for near-real portfolios and markets used by the integration suite.

A book of Brazilian CDI-vs-IPCA swaps: receive CDI (id_indexador 1), pay an
IPCA-linked leg (id_indexador 2), spread across a realistic tenor grid and a
range of notionals. The market carries a full DI/IPCA curve, an IPCA fixing, and
projected index points so every leg's market join resolves.
"""

from __future__ import annotations

from datetime import date
from itertools import cycle, islice
from typing import cast

import polars as pl
from pandera.typing.polars import LazyFrame
from schenberg.domain.schemas import SwapInput
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource

AS_OF = date(2026, 6, 3)

# Business-day tenors out to ~4y on a 252 calendar.
TENORS = (21, 63, 126, 252, 504, 756, 1008)
NOTIONALS = (1_000_000.0, 2_500_000.0, 5_000_000.0, 750_000.0, 10_000_000.0)

CDI = 1  # id_indexador for the floating CDI curve
IPCA = 2  # id_indexador for the inflation-linked leg


def make_market(*, zero_rate: float = 0.10, slope: float = 0.005) -> MarketSnapshot:
    """A DI/IPCA market. ``zero_rate`` sets the short end; ``slope`` adds a gentle
    upward term structure per tenor bucket so discounting is non-trivial."""
    rows = []
    for idx in (CDI, IPCA):
        for i, tenor in enumerate(TENORS):
            rows.append(
                {
                    "id_indexador": idx,
                    "tenor_days": tenor,
                    "zero_rate": zero_rate + slope * i,
                    # CDI projection lives on the curve; IPCA leg doesn't use it.
                    "forward_rate": (zero_rate + 0.02 + slope * i) if idx == CDI else None,
                }
            )
    curves = pl.DataFrame(rows).lazy()

    fixings = pl.DataFrame(
        {"id_indexador": [IPCA], "fixing_date": [AS_OF], "fixing_value": [100.0]}
    ).lazy()

    projected = pl.DataFrame(
        {
            "id_indexador": [IPCA] * len(TENORS),
            "tenor_days": list(TENORS),
            # ~5% annual inflation compounded over the tenor.
            "projected_index": [100.0 * (1.05 ** (t / 252.0)) for t in TENORS],
        }
    ).lazy()

    return MarketSnapshot.from_sources(
        as_of=AS_OF,
        sources=[
            MarketSource("curves", curves),
            MarketSource("fixings", fixings),
            MarketSource("projected_indexes", projected),
        ],
    )


def make_swaps(n: int) -> LazyFrame[SwapInput]:
    """``n`` CDI-vs-IPCA swaps, deterministically spread across tenors/notionals."""
    tenors = list(islice(cycle(TENORS), n))
    notionals = list(islice(cycle(NOTIONALS), n))
    return cast(
        LazyFrame[SwapInput],
        pl.DataFrame(
            {
                "swap_id": [f"SWP-{i:06d}" for i in range(n)],
                "notional": notionals,
                "id_indexador_ativo": [CDI] * n,
                "id_indexador_passivo": [IPCA] * n,
                "indexador_kind_ativo": ["CDI"] * n,
                "indexador_kind_passivo": ["IPCA"] * n,
                "payment_days": tenors,
                "accrual": [t / 252.0 for t in tenors],
                "base_date": [AS_OF] * n,
                "fixed_rate_ativo": [None] * n,
                "fixed_rate_passivo": [None] * n,
                "real_coupon_ativo": [None] * n,
                "real_coupon_passivo": [0.02] * n,
            }
        ).lazy(),
    )


def make_positions(swap_ids: list[str]) -> pl.LazyFrame:
    """Spread the catalog across three books, with a couple of swaps double-booked
    (so the valuer's price-once-then-join is exercised)."""
    books = cycle(["Rates", "Inflation", "Macro"])
    quantities = cycle([1.0, 2.0, 0.5, 3.0])
    rows = [
        {
            "position_id": f"POS-{i:04d}",
            "book": book,
            "swap_id": swap_id,
            "quantity": qty,
        }
        for i, (swap_id, book, qty) in enumerate(zip(swap_ids, books, quantities, strict=False))
    ]
    # double-book the first swap into a second book to test aggregation
    rows.append(
        {"position_id": "POS-DUP", "book": "Macro", "swap_id": swap_ids[0], "quantity": 1.5}
    )
    return pl.DataFrame(rows).lazy()
