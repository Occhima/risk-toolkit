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


class BuySell(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class SwapLegKind(StrEnum):
    FIXED = "FIXED"
    CDI = "CDI"
    IPCA = "IPCA"


class PayReceive(StrEnum):
    PAY = "PAY"
    RECEIVE = "RECEIVE"
