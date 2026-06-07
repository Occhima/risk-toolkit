"""Small semantic market-data DSL that builds :class:`MarketRole` objects.

This module is only sugar over ``market_role(...).read(...).by(...).fixing(...)``.
It does not interact with ``FormulaGraph``; roles are still resolved by ``bind``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from schenberg.market_data.roles import Fixing, MarketRole, market_role


@dataclass(frozen=True, slots=True)
class SemanticRole:
    """Fluent role builder used by ``CURVES``, ``FIXINGS`` and ``VOLS``."""

    role: MarketRole
    default_source: str | None = None

    @property
    def name(self) -> str:
        return self.role.name

    @property
    def source_name(self) -> str | None:
        return self.role.source

    @property
    def exact(self) -> Any:
        return self.role.exact

    def source(self, name: str) -> MarketRole:
        if self.role.value_col is None:
            raise ValueError(f"semantic role {self.role.name!r} has no value column")
        return self.role.read(name, self.role.value_col)

    def by(self, **bindings: str) -> SemanticRole:
        return SemanticRole(self.role.by(**bindings), self.default_source)

    def for_tenor(self, contract_col: str, quote_col: str | None = None) -> SemanticRole:
        return self.by(**{contract_col: quote_col or "tenor_days"})

    def for_expiry(self, contract_col: str, quote_col: str | None = None) -> SemanticRole:
        return self.by(**{contract_col: quote_col or "expiry"})

    def for_strike(self, contract_col: str, quote_col: str | None = None) -> SemanticRole:
        return self.by(**{contract_col: quote_col or "strike"})

    def fixing(self, quote_col: str, rule: Fixing | pl.Expr) -> MarketRole:
        return self.role.fixing(quote_col, rule)

    def to_role(self) -> MarketRole:
        return self.role


@dataclass(frozen=True, slots=True)
class _Curves:
    def zero_rate(self, curve: str | None = None, *, as_: str = "zero_rate") -> SemanticRole:
        role = market_role(as_).read("curves", "zero_rate")
        if curve is not None:
            role = role.by(curve="curve")
        return SemanticRole(role)

    def discount_factor(
        self, curve: str | None = None, *, as_: str = "discount_factor"
    ) -> SemanticRole:
        role = market_role(as_).read("curves", "discount_factor")
        if curve is not None:
            role = role.by(curve="curve")
        return SemanticRole(role)

    factor = discount_factor


@dataclass(frozen=True, slots=True)
class _Fixings:
    def value(self, name: str | None = None, *, as_: str = "fixing") -> SemanticRole:
        role = market_role(as_).read("fixings", "value")
        if name is not None:
            role = role.by(currency_pair="currency_pair")
        return SemanticRole(role)


@dataclass(frozen=True, slots=True)
class _Vols:
    def implied(self, name: str | None = None, *, as_: str = "vol") -> SemanticRole:
        role = market_role(as_).read("vol_surface", "implied_vol")
        if name is not None:
            role = role.by(currency_pair="currency_pair")
        return SemanticRole(role)


CURVES = _Curves()
FIXINGS = _Fixings()
VOLS = _Vols()

__all__ = ["CURVES", "FIXINGS", "VOLS", "SemanticRole"]
