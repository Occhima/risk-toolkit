"""Domain enum constants used at public pricing boundaries."""

from __future__ import annotations

from enum import StrEnum


class InstrumentType(StrEnum):
    FORWARD = "FORWARD"
    SWAP = "SWAP"
    OPTION = "OPTION"


class ForwardFamily(StrEnum):
    GENERIC = "GENERIC"
    ENERGY = "ENERGY"
    FX = "FX"
    EQUITY_INDEX = "EQUITY_INDEX"


class SettlementType(StrEnum):
    PHYSICAL = "PHYSICAL"
    CASH_SETTLED = "CASH_SETTLED"


class OptionModel(StrEnum):
    """How the cost of carry ``b`` in the generalized BSM formula is formed.

    GENERALIZED takes ``b`` straight from market data (the carry knob);
    MERTON derives it from a dividend yield as ``b = r - q``.
    """

    GENERALIZED = "GENERALIZED"
    MERTON = "MERTON"


class OptionKind(StrEnum):
    CALL = "CALL"
    PUT = "PUT"


class GreekMethod(StrEnum):
    ANALYTIC = "ANALYTIC"  # closed-form partials
    NUMERIC = "NUMERIC"  # central finite differences
    AUTODIFF = "AUTODIFF"  # autograd reverse-mode AD


class SwapLegKind(StrEnum):
    FIXED = "FIXED"
    CDI = "CDI"
    IPCA = "IPCA"


class PayReceive(StrEnum):
    PAY = "PAY"
    RECEIVE = "RECEIVE"


class AccrualConvention(StrEnum):
    COMPOUND = "COMPOUND"
    CONTINUOUS = "CONTINUOUS"
    LINEAR = "LINEAR"
