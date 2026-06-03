"""Swap-level <-> leg-level transforms. Pure, lazy, no collect."""
from __future__ import annotations

import polars as pl

_COMMON = ["swap_id", "notional", "payment_days", "accrual", "base_date"]


def swap_to_legs(swaps: pl.LazyFrame) -> pl.LazyFrame:
    """Wide swap rows -> long leg rows. ativo: pay_receive=+1, passivo: -1."""
    def leg(side: str, sign: float) -> pl.LazyFrame:
        return swaps.select(
            *_COMMON,
            pl.lit(side).alias("leg"),
            pl.lit(sign).alias("pay_receive"),
            pl.col(f"id_indexador_{side}").alias("id_indexador"),
            pl.col(f"indexador_kind_{side}").alias("indexador_kind"),
            pl.col(f"fixed_rate_{side}").alias("fixed_rate"),
            pl.col(f"real_coupon_{side}").alias("real_coupon"),
        )
    return pl.concat([leg("ativo", 1.0), leg("passivo", -1.0)], how="vertical")


def aggregate_swap_pv(legs: pl.LazyFrame) -> pl.LazyFrame:
    """Leg-level PVs -> swap-level npv. `pv` already carries the pay_receive
    sign, so npv is a plain sum and the per-leg columns are signed
    contributions. Do not re-apply the sign here."""
    return legs.group_by("swap_id").agg(
        npv=pl.col("pv").sum(),
        ativo_pv=pl.col("pv").filter(pl.col("leg") == "ativo").sum(),
        passivo_pv=pl.col("pv").filter(pl.col("leg") == "passivo").sum(),
    )
