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
from schenberg.domain.schemas import SwapLegInput
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


def make_swap_legs(n: int) -> LazyFrame[SwapLegInput]:
    """``n`` CDI-vs-IPCA swaps as normalized legs (receive CDI / pay IPCA),
    two rows per swap, deterministically spread across tenors/notionals."""
    swap_ids = [f"SWP-{i:06d}" for i in range(n)]
    tenors = list(islice(cycle(TENORS), n))
    notionals = list(islice(cycle(NOTIONALS), n))
    accruals = [t / 252.0 for t in tenors]

    def side(
        leg_id: str,
        leg_weight: float,
        indexer: int,
        kind: str,
        real_coupon: float | None,
    ) -> dict[str, list]:
        return {
            "swap_id": swap_ids,
            "leg_id": [leg_id] * n,
            "leg_kind": [kind] * n,
            "leg_role": [leg_id] * n,
            "leg_weight": [leg_weight] * n,
            "notional": notionals,
            "id_indexador": [indexer] * n,
            "payment_days": tenors,
            "accrual": accruals,
            "base_date": [AS_OF] * n,
            "fixed_rate": [None] * n,
            "real_coupon": [real_coupon] * n,
            "cashflow_amount": [None] * n,
        }

    floats = {"fixed_rate": pl.Float64, "real_coupon": pl.Float64, "cashflow_amount": pl.Float64}
    ativo = pl.DataFrame(side("ativo", 1.0, CDI, "CDI", None), schema_overrides=floats)
    passivo = pl.DataFrame(side("passivo", -1.0, IPCA, "IPCA", 0.02), schema_overrides=floats)
    return cast(LazyFrame[SwapLegInput], pl.concat([ativo, passivo]).lazy())
