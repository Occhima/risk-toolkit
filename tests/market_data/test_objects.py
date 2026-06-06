from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from schenberg.market_data.calendar import ANBIMA
from schenberg.market_data.calendar.conventions import Calendar
from schenberg.market_data.objects import (
    Compounding,
    CompoundingKind,
    CurveConvention,
    CurvePoint,
    ForwardCurve,
    MissingMarketSourceError,
    QuoteKind,
    VolatilityConvention,
    VolatilityPoint,
    VolatilitySurface,
    VolQuoteKind,
)
from schenberg.market_data.roles import market_role
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource

REF_DATE = date(2026, 1, 2)
TENOR = date(2026, 12, 30)
EXPIRY = date(2026, 6, 30)
_RTOL = 1e-8
_ATOL = 1e-10

EXPO_CONV = CurveConvention(
    calendar=ANBIMA,
    compounding=Compounding(CompoundingKind.EXPONENTIAL),
    quote_kind=QuoteKind.RATE,
)
LINEAR_CONV = CurveConvention(
    calendar=ANBIMA,
    compounding=Compounding(CompoundingKind.LINEAR),
    quote_kind=QuoteKind.RATE,
)
DF_CONV = CurveConvention(
    calendar=ANBIMA,
    compounding=Compounding(CompoundingKind.DISCOUNT_FACTOR),
    quote_kind=QuoteKind.FACTOR,
)
VOL_CONV_STRIKE = VolatilityConvention(axes=("expiry", "strike"), quote_kind=VolQuoteKind.LOGNORMAL)
VOL_CONV_MONEYNESS = VolatilityConvention(
    axes=("expiry", "moneyness"), quote_kind=VolQuoteKind.LOGNORMAL
)


def _rate_frame() -> pl.DataFrame:
    return pl.DataFrame({"tenor": [TENOR], "rate": [0.12]})


def _factor_frame() -> pl.DataFrame:
    return pl.DataFrame({"tenor": [TENOR], "factor": [0.887]})


def _vol_frame_strike() -> pl.DataFrame:
    return pl.DataFrame({"expiry": [EXPIRY], "strike": [5000.0], "volatility": [0.22]})


def _vol_frame_moneyness() -> pl.DataFrame:
    return pl.DataFrame({"expiry": [EXPIRY], "moneyness": [1.05], "volatility": [0.22]})


# ---------------------------------------------------------------------------
# 1. from_frame accepts eager pl.DataFrame
# ---------------------------------------------------------------------------


def test_forward_curve_from_eager_frame():
    curve = ForwardCurve.from_frame(
        _rate_frame(), name="di_curve", ref_date=REF_DATE, convention=EXPO_CONV
    )
    assert isinstance(curve.data, pl.DataFrame)
    assert "curve" in curve.data.columns
    assert curve.data["curve"][0] == "di_curve"


# ---------------------------------------------------------------------------
# 2. from_frame accepts lazy pl.LazyFrame
# ---------------------------------------------------------------------------


def test_forward_curve_from_lazy_frame():
    curve = ForwardCurve.from_frame(
        _rate_frame().lazy(), name="di_curve", ref_date=REF_DATE, convention=EXPO_CONV
    )
    assert isinstance(curve.data, pl.DataFrame)


# ---------------------------------------------------------------------------
# 3. to_market_source returns MarketSource with schema=CurvePoint
# ---------------------------------------------------------------------------


def test_forward_curve_to_market_source_type():
    curve = ForwardCurve.from_frame(
        _rate_frame(), name="di_curve", ref_date=REF_DATE, convention=EXPO_CONV
    )
    ms = curve.to_market_source()
    assert isinstance(ms, MarketSource)
    assert ms.schema is CurvePoint
    assert ms.name == "di_curve"


# ---------------------------------------------------------------------------
# 4. data stays DataFrame; to_market_source().data is LazyFrame
# ---------------------------------------------------------------------------


def test_forward_curve_laziness_at_edge():
    curve = ForwardCurve.from_frame(
        _rate_frame(), name="di_curve", ref_date=REF_DATE, convention=EXPO_CONV
    )
    assert isinstance(curve.data, pl.DataFrame)
    assert isinstance(curve.to_market_source().data, pl.LazyFrame)


# ---------------------------------------------------------------------------
# 5. Exponential vs linear compounding produce different factors
# ---------------------------------------------------------------------------


def test_exponential_vs_linear_factor():
    expo = ForwardCurve.from_frame(_rate_frame(), name="c", ref_date=REF_DATE, convention=EXPO_CONV)
    lin = ForwardCurve.from_frame(
        _rate_frame(), name="c", ref_date=REF_DATE, convention=LINEAR_CONV
    )
    f_expo = expo.data["factor"][0]
    f_lin = lin.data["factor"][0]
    assert f_expo is not None and f_lin is not None
    assert abs(f_expo - f_lin) > _RTOL


# ---------------------------------------------------------------------------
# 6. FACTOR-quoted curve: factor column passthrough, not recomputed
# ---------------------------------------------------------------------------


def test_discount_factor_passthrough():
    original_factor = 0.887
    curve = ForwardCurve.from_frame(
        _factor_frame(), name="c", ref_date=REF_DATE, convention=DF_CONV
    )
    assert abs(curve.data["factor"][0] - original_factor) < _ATOL


# ---------------------------------------------------------------------------
# 7. Calendar/day-count logic lives in Calendar, not ForwardCurve
# ---------------------------------------------------------------------------


def test_calendar_drives_business_days():
    no_holiday_cal = Calendar(
        business_days_per_year=252,
        accrual=ANBIMA.accrual,
        holidays=frozenset(),
        name="no-hol",
    )
    conv_no_hol = CurveConvention(
        calendar=no_holiday_cal,
        compounding=Compounding(CompoundingKind.EXPONENTIAL),
        quote_kind=QuoteKind.RATE,
    )
    curve_anbima = ForwardCurve.from_frame(
        _rate_frame(), name="c", ref_date=REF_DATE, convention=EXPO_CONV
    )
    curve_nohol = ForwardCurve.from_frame(
        _rate_frame(), name="c", ref_date=REF_DATE, convention=conv_no_hol
    )
    bd_anbima = curve_anbima.data["business_days"][0]
    bd_nohol = curve_nohol.data["business_days"][0]
    assert bd_nohol >= bd_anbima


# ---------------------------------------------------------------------------
# 8. VolatilitySurface.from_frame works, stamps surface column
# ---------------------------------------------------------------------------


def test_vol_surface_from_frame():
    vol = VolatilitySurface.from_frame(
        _vol_frame_strike(), name="vol_surface", ref_date=REF_DATE, convention=VOL_CONV_STRIKE
    )
    assert isinstance(vol.data, pl.DataFrame)
    assert "surface" in vol.data.columns
    assert vol.data["surface"][0] == "vol_surface"


# ---------------------------------------------------------------------------
# 9. VolatilitySurface.to_market_source returns MarketSource with schema
# ---------------------------------------------------------------------------


def test_vol_surface_to_market_source():
    vol = VolatilitySurface.from_frame(
        _vol_frame_strike(), name="vol_surface", ref_date=REF_DATE, convention=VOL_CONV_STRIKE
    )
    ms = vol.to_market_source()
    assert isinstance(ms, MarketSource)
    assert ms.schema is VolatilityPoint
    assert isinstance(ms.data, pl.LazyFrame)


# ---------------------------------------------------------------------------
# 10. Vol schema supports expiry×strike and expiry×moneyness
# ---------------------------------------------------------------------------


def test_vol_surface_strike_and_moneyness_axes():
    vol_strike = VolatilitySurface.from_frame(
        _vol_frame_strike(), name="vs", ref_date=REF_DATE, convention=VOL_CONV_STRIKE
    )
    vol_moneyness = VolatilitySurface.from_frame(
        _vol_frame_moneyness(), name="vm", ref_date=REF_DATE, convention=VOL_CONV_MONEYNESS
    )
    assert "strike" in vol_strike.data.columns
    assert "moneyness" in vol_moneyness.data.columns
    assert vol_strike.data["moneyness"][0] is None
    assert vol_moneyness.data["strike"][0] is None


# ---------------------------------------------------------------------------
# 11. MarketSnapshot built from both curve and vol sources
# ---------------------------------------------------------------------------


def test_market_snapshot_from_curve_and_vol():
    curve = ForwardCurve.from_frame(
        _rate_frame(), name="di_curve", ref_date=REF_DATE, convention=EXPO_CONV
    )
    vol = VolatilitySurface.from_frame(
        _vol_frame_strike(), name="vol_surface", ref_date=REF_DATE, convention=VOL_CONV_STRIKE
    )
    snapshot = MarketSnapshot.from_sources(
        as_of=REF_DATE,
        sources=[curve.to_market_source(), vol.to_market_source()],
    )
    assert isinstance(snapshot.source("di_curve"), MarketSource)
    assert isinstance(snapshot.source("vol_surface"), MarketSource)


# ---------------------------------------------------------------------------
# 12. MarketRequirements bind from canonical sources
# ---------------------------------------------------------------------------


def test_market_role_binds_from_canonical_source():
    curve = ForwardCurve.from_frame(
        _rate_frame(), name="di_curve", ref_date=REF_DATE, convention=EXPO_CONV
    )
    snapshot = MarketSnapshot.from_sources(as_of=REF_DATE, sources=[curve.to_market_source()])

    role = market_role("rate").read("di_curve", "rate").by(tenor="tenor")

    input_lf = pl.DataFrame({"tenor": [TENOR]}).lazy()
    result = role.attach(input_lf, snapshot).collect()
    assert "rate" in result.columns
    assert result["rate"][0] is not None


# ---------------------------------------------------------------------------
# 13. Missing source raises MissingMarketSourceError
# ---------------------------------------------------------------------------


def test_missing_source_raises_custom_error():
    snapshot = MarketSnapshot.from_sources(as_of=REF_DATE, sources=[])
    with pytest.raises(MissingMarketSourceError) as exc_info:
        snapshot.source("nonexistent")
    err = exc_info.value
    assert isinstance(err, KeyError)
    assert isinstance(err, ValueError)
    assert "nonexistent" in str(err)


# ---------------------------------------------------------------------------
# 14. to_market_source() wraps data in LazyFrame without materializing it
# ---------------------------------------------------------------------------


def test_to_market_source_returns_lazy_no_collect():
    curve = ForwardCurve.from_frame(
        _rate_frame(), name="di_curve", ref_date=REF_DATE, convention=EXPO_CONV
    )
    ms = curve.to_market_source()

    assert isinstance(ms.data, pl.LazyFrame)

    collected = ms.data.collect()
    assert collected.shape == curve.data.shape
