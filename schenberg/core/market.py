"""Declarative market requirements and a lazy market-data container.

Graphs declare what market data they need; formulas never perform lookups.
A MarketRequirement is a left-join spec against one named MarketSnapshot frame:
left_keys (on the leg frame) <-> right_keys (on the market frame), renaming the
requested source columns to their attached names. All injection stays lazy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl


@dataclass(frozen=True, slots=True)
class MarketRequirement:
    table: str  # which MarketSnapshot frame to join
    left_keys: tuple[str, ...]  # join keys on the leg frame
    right_keys: tuple[str, ...]  # join keys on the market frame
    outputs: dict[str, str]  # source_column -> attached_column (renamed on join)


# --- constructors: defaults absorb the conventional keys -------------------
def curve(
    *identity: str,
    indexer_col: str = "id_indexador",
    tenor_col: str = "payment_days",
    outputs: dict[str, str] | None = None,
) -> MarketRequirement:
    """Positional args are identity outputs: curve("zero_rate", "forward_rate")."""
    out = {n: n for n in identity}
    if outputs:
        out.update(outputs)
    return MarketRequirement("curves", (indexer_col, tenor_col), (indexer_col, "tenor_days"), out)


def fixing(
    *, indexer_col: str = "id_indexador", date_col: str = "base_date", output: str = "base_index"
) -> MarketRequirement:
    return MarketRequirement(
        "fixings", (indexer_col, date_col), (indexer_col, "fixing_date"), {"fixing_value": output}
    )


def projected_index(
    *,
    indexer_col: str = "id_indexador",
    tenor_col: str = "payment_days",
    output: str = "projected_index",
) -> MarketRequirement:
    return MarketRequirement(
        "projected_indexes",
        (indexer_col, tenor_col),
        (indexer_col, "tenor_days"),
        {"projected_index": output},
    )


def energy_forward(
    *,
    submarket_col: str = "submarket",
    period_col: str = "delivery_period",
    outputs: dict[str, str] | None = None,
) -> MarketRequirement:
    """Power forward curve keyed by (submarket, delivery_period). By default brings
    forward_price and settle_days (renamed to payment_days, the discount tenor)."""
    out = outputs or {"forward_price": "forward_price", "settle_days": "payment_days"}
    return MarketRequirement(
        "forward_curves", (submarket_col, period_col), ("submarket", "delivery_period"), out
    )


def fx(*, currency_col: str = "currency", output: str = "fx_rate") -> MarketRequirement:
    """FX rate keyed by currency (local -> reporting)."""
    return MarketRequirement("fx_rates", (currency_col,), ("currency",), {"fx_rate": output})


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    as_of: date
    curves: pl.LazyFrame
    fixings: pl.LazyFrame | None = None
    projected_indexes: pl.LazyFrame | None = None
    forward_curves: pl.LazyFrame | None = None
    fx_rates: pl.LazyFrame | None = None  # currency, fx_rate

    def attach(self, lf: pl.LazyFrame, req: MarketRequirement) -> pl.LazyFrame:
        # One generic left join for every requirement kind. Extension point for
        # interpolation: swap the exact join for join_asof on the tenor key.
        src = getattr(self, req.table, None)
        if src is None:
            raise ValueError(f"snapshot has no {req.table!r} frame for this requirement")
        right = src.select([*req.right_keys, *req.outputs]).rename(req.outputs)
        return lf.join(
            right,
            left_on=list(req.left_keys),
            right_on=list(req.right_keys),
            how="left",
        )
