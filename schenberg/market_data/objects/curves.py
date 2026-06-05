from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import pandera.polars as pa
import polars as pl

from schenberg.domain.base import SchenbergDataFrameModel
from schenberg.market_data.objects.conventions import CurveConvention, QuoteKind
from schenberg.market_data.sources import MarketSource


class CurvePoint(SchenbergDataFrameModel):
    """Canonical schema for a single normalized curve quote."""

    curve: str
    ref_date: date = pa.Field(nullable=True)
    tenor: date
    business_days: int = pa.Field(nullable=True)
    year_fraction: float = pa.Field(nullable=True)
    rate: float = pa.Field(nullable=True)
    factor: float = pa.Field(nullable=True)
    unit_price: float = pa.Field(nullable=True)


@dataclass(frozen=True, slots=True)
class ForwardCurve:
    """Reusable market-data object for an interest-rate / forward curve.

    Wraps a lazy frame of raw quotes and a :class:`CurveConvention`. The
    object itself holds no financial maths: ``normalize`` calls into the
    convention's :meth:`CurveConvention.normalize_exprs` lazily. Exporting
    to a :class:`MarketSource` is a pure metadata operation - no ``.collect()``
    is ever called.
    """

    name: str
    ref_date: date
    data: pl.LazyFrame
    convention: CurveConvention

    @classmethod
    def from_frame(
        cls,
        raw: pl.DataFrame | pl.LazyFrame,
        *,
        name: str,
        ref_date: date,
        convention: CurveConvention,
        ref_date_col: str = "ref_date",
        tenor_col: str = "tenor",
        normalize: bool = True,
    ) -> ForwardCurve:
        """Build a normalized :class:`ForwardCurve` from a raw eager or lazy frame.

        The input may be a :class:`polars.DataFrame` or
        :class:`polars.LazyFrame`. Internally everything is treated lazily.
        When ``normalize=True`` (the default) the convention's expressions
        are applied immediately so the resulting ``data`` already conforms
        to the canonical :class:`CurvePoint` schema.
        """
        lf = raw.lazy() if isinstance(raw, pl.DataFrame) else raw
        curve = cls(name=name, ref_date=ref_date, data=lf, convention=convention)
        if normalize:
            curve = curve.normalize(ref_date_col=ref_date_col, tenor_col=tenor_col)
        return curve

    def normalize(
        self,
        *,
        ref_date_col: str = "ref_date",
        tenor_col: str = "tenor",
    ) -> ForwardCurve:
        """Return a new :class:`ForwardCurve` with canonical columns added.

        Adds ``ref_date`` (if missing), ``curve`` (the curve name), and the
        expressions produced by the convention. The original input columns
        are preserved so vendor-supplied aliases (e.g. ``zero_rate``,
        ``tenor_days``) stay available downstream.
        """
        schema_names = self.data.collect_schema().names()
        prep: list[pl.Expr] = []
        if ref_date_col not in schema_names:
            prep.append(pl.lit(self.ref_date).alias(ref_date_col))
        prep.append(pl.lit(self.name).alias("curve"))
        prepped = self.data.with_columns(prep)
        normalized = prepped.with_columns(
            self.convention.normalize_exprs(
                ref_date_col=ref_date_col, tenor_col=tenor_col
            )
        )
        return replace(self, data=normalized)

    def to_market_source(self) -> MarketSource:
        """Export this curve as a canonical :class:`MarketSource`."""
        return MarketSource(name=self.name, data=self.data, schema=CurvePoint)

    def with_legacy_aliases(
        self,
        *,
        indexer_id: int,
        rate_alias: str = "zero_rate",
        tenor_days_alias: str = "tenor_days",
        indexer_col: str = "id_indexador",
    ) -> ForwardCurve:
        """Opt-in helper: emit legacy alias columns for existing curve specs.

        The existing :data:`schenberg.pricing.market.DI` / ``CURVES`` specs
        join on ``id_indexador`` + ``tenor_days`` and read ``zero_rate``.
        Pricing code written against those constants keeps working when the
        canonical curve also exposes those names. This is purely additive -
        the canonical columns are not removed.
        """
        aliases = [
            pl.lit(indexer_id).alias(indexer_col),
            pl.col("business_days").alias(tenor_days_alias),
        ]
        if self.convention.quote_kind is QuoteKind.RATE:
            aliases.append(pl.col("rate").alias(rate_alias))
        elif self.convention.quote_kind is QuoteKind.FACTOR:
            aliases.append(pl.col("rate").alias(rate_alias))
        with_aliases = self.data.with_columns(aliases)
        return replace(self, data=with_aliases)
