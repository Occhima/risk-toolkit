"""Market roles: the semantic unit that glues a raw contract to its market data.

A :class:`MarketRole` is a self-contained description of *one* market column a
pricing input needs: which source to read, which quote value, how to find the row
(exact keys plus an optional date :class:`Fixing` derived from the contract), and
the single column it publishes into the enriched input schema. Roles never live
inside the pricing graph — they are resolved *before* it, by :func:`bind`, so the
graph stays a pure function of its input columns.

    ForwardPrice = (
        market_role("forward_price")
        .read("curves", "forward_rate")
        .by(indexer="id_indexador", payment_days="tenor_days")
    )

    Indexer = (
        market_role("indexer_value")
        .read("fixings", "fixing_value")
        .by(indexer="id_indexador")
        .fixing("fixing_date", same_day("tenor"))
    )

The published column is the role *name*. The join is executed by the proven
:class:`~schenberg.core.market.MarketRequirement`; the role only adds the fixing
derivation and the schema-mixin generation (:class:`With`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import pandera.polars as pa
import polars as pl

from schenberg.core.columns import ColumnBinding, ColumnSet
from schenberg.core.market import MarketRequirement
from schenberg.domain.base import SchenbergDataFrameModel

if TYPE_CHECKING:
    from schenberg.market_data.snapshot import MarketSnapshot


@dataclass(frozen=True, slots=True)
class Fixing:
    """Derives a date join-key column from contract columns (typically ``tenor``).

    Two forms:

    * **Direct** — one rule for every row::

        Fixing.rule(same_day("tenor"))

    * **Conditional** — branch on a selector column, mirroring contract fixing
      conventions (CPI fixes 5 days out, everything else same-day)::

        (Fixing.on("indexer")
            .when("CPI", add_days("tenor", 5))
            .otherwise(same_day("tenor")))

    The branch/default expressions are the un-aliased :mod:`date_rules` helpers.
    :meth:`expr` compiles to a single Polars expression.
    """

    selector: str | None
    branches: tuple[tuple[str, pl.Expr], ...]
    default: pl.Expr | None

    @classmethod
    def rule(cls, expr: pl.Expr) -> Fixing:
        """A single, unconditional fixing rule."""
        return cls(selector=None, branches=(), default=expr)

    @classmethod
    def on(cls, selector: str) -> Fixing:
        """Start a conditional fixing keyed on ``selector`` (a contract column)."""
        return cls(selector=selector, branches=(), default=None)

    def when(self, value: str, expr: pl.Expr) -> Fixing:
        """Add a branch: when ``selector == value`` use ``expr``."""
        if self.selector is None:
            raise ValueError("Fixing.when(...) requires Fixing.on(selector) first")
        return replace(self, branches=(*self.branches, (value, expr)))

    def otherwise(self, expr: pl.Expr) -> Fixing:
        """Set the fallback rule for a conditional fixing."""
        return replace(self, default=expr)

    def expr(self) -> pl.Expr:
        if self.selector is None:
            if self.default is None:
                raise ValueError("empty Fixing: use Fixing.rule(...) or Fixing.on(...)")
            return self.default
        if not self.branches:
            raise ValueError("conditional Fixing has no .when(...) branches")
        if self.default is None:
            raise ValueError("conditional Fixing needs a .otherwise(...) fallback")
        (first_value, first_expr), *rest = self.branches
        chain = pl.when(pl.col(self.selector) == first_value).then(first_expr)
        for value, expr in rest:
            chain = chain.when(pl.col(self.selector) == value).then(expr)
        return chain.otherwise(self.default)


@dataclass(frozen=True, slots=True)
class LiteralBinding:
    """A quote-side key matched against a literal value via a transient left column."""

    right: str
    value: object
    temp_left: str


@dataclass(frozen=True, slots=True)
class MarketRole:
    """One resolvable market column: read + keys + optional fixing + published name."""

    name: str  # the column this role publishes into the pricing input
    source: str | None = None  # snapshot source / table name
    value_col: str | None = None  # quote-side column to read
    exact: tuple[ColumnBinding, ...] = ()  # contract column -> quote column
    literal: tuple[LiteralBinding, ...] = ()  # literal value -> quote column
    fixing_quote: str | None = None  # quote-side date column the fixing matches
    fixing_rule: Fixing | None = None

    def read(self, source: str, value_col: str) -> MarketRole:
        """Select the source table and the quote column to read from it."""
        return replace(self, source=source, value_col=value_col)

    def by(self, **bindings: str) -> MarketRole:
        """Declare exact join keys as ``contract_column=quote_column`` pairs."""
        exact = tuple(ColumnBinding(left, right) for left, right in bindings.items())
        return replace(self, exact=self.exact + exact)

    def by_literal(self, **bindings: object) -> MarketRole:
        """Declare quote-side join keys matched against literal values.

        Literal values are added as transient left columns so the existing lazy
        market join path can resolve constants without requiring contract rows to
        carry redundant key columns.
        """
        literal = tuple(
            LiteralBinding(
                right=quote_col,
                value=value,
                temp_left=f"__const_{self.name}_{quote_col}",
            )
            for quote_col, value in bindings.items()
        )
        return replace(self, literal=self.literal + literal)

    def fixing(self, quote_col: str, rule: Fixing | pl.Expr) -> MarketRole:
        """Add a derived date key: ``rule`` (over contract columns) matches the
        quote-side ``quote_col``. ``rule`` may be a bare expression (wrapped as a
        direct :class:`Fixing`) or a conditional :class:`Fixing`."""
        fix = rule if isinstance(rule, Fixing) else Fixing.rule(rule)
        return replace(self, fixing_quote=quote_col, fixing_rule=fix)

    def for_tenor(self, contract_col: str, quote_col: str | None = None) -> MarketRole:
        """Semantic alias for a tenor join key."""
        return self.by(**{contract_col: quote_col or "tenor_days"})

    def for_expiry(self, contract_col: str, quote_col: str | None = None) -> MarketRole:
        """Semantic alias for an expiry join key."""
        return self.by(**{contract_col: quote_col or "expiry"})

    def for_strike(self, contract_col: str, quote_col: str | None = None) -> MarketRole:
        """Semantic alias for a strike join key."""
        return self.by(**{contract_col: quote_col or "strike"})

    def _requirement(self, bindings: tuple[ColumnBinding, ...]) -> MarketRequirement:
        if self.source is None or self.value_col is None:
            raise ValueError(f"role {self.name!r} has no .read(source, value_col)")
        return MarketRequirement(
            table=self.source,
            on=ColumnSet(bindings),
            outputs={self.value_col: self.name},
        )

    def attach(self, lf: pl.LazyFrame, snapshot: MarketSnapshot) -> pl.LazyFrame:
        """Resolve this role against the snapshot: derive the fixing key (if any),
        left-join the source, publish the value as ``name``, drop the transient
        fixing column. Stays lazy."""
        bindings = self.exact
        drop_cols = [literal.temp_left for literal in self.literal]
        for literal in self.literal:
            lf = lf.with_columns(pl.lit(literal.value).alias(literal.temp_left))
            bindings = bindings + (ColumnBinding(literal.temp_left, literal.right),)
        fix_col: str | None = None
        if self.fixing_rule is not None:
            if self.fixing_quote is None:
                raise ValueError(f"role {self.name!r} fixing has no quote column")
            fix_col = f"__fix_{self.name}"
            lf = lf.with_columns(self.fixing_rule.expr().alias(fix_col))
            bindings = bindings + (ColumnBinding(fix_col, self.fixing_quote),)
            drop_cols.append(fix_col)
        out = self._requirement(bindings).attach(lf, snapshot)
        return out.drop(drop_cols) if drop_cols else out


def market_role(name: str) -> MarketRole:
    """Start declaring a market role that publishes the column ``name``."""
    return MarketRole(name=name)


# ---- schema mixins -----------------------------------------------------------

_ROLE_ATTR = "__market_role__"


class With:
    """``With[role]`` -> a Pandera mixin declaring the float column ``role`` adds.

    Compose them onto a pricing-input schema; :func:`roles_of` discovers them so
    :func:`bind` knows what to join. Market columns are *resolved* data, never
    user-supplied — the schema declares them, the user never passes them."""

    def __class_getitem__(cls, role: MarketRole) -> type[pa.DataFrameModel]:
        to_role = getattr(role, "to_role", None)
        if callable(to_role):
            role = to_role()
        namespace: dict[str, Any] = {
            "__annotations__": {role.name: float},
            "__module__": __name__,
            _ROLE_ATTR: role,
        }
        return type(f"With_{role.name}", (SchenbergDataFrameModel,), namespace)


def roles_of(schema: type[Any]) -> list[MarketRole]:
    """The market roles declared on ``schema`` via ``With[...]``, in MRO order."""
    roles: list[MarketRole] = []
    seen: set[str] = set()
    for klass in schema.__mro__:
        role = klass.__dict__.get(_ROLE_ATTR)
        if isinstance(role, MarketRole) and role.name not in seen:
            seen.add(role.name)
            roles.append(role)
    return roles


# ---- the bind boundary -------------------------------------------------------


def bind(
    raw: pl.LazyFrame | pl.DataFrame,
    snapshot: MarketSnapshot,
    schema: type[Any],
) -> pl.LazyFrame:
    """Glue a raw contract to its market data, producing an enriched pricing input.

    Discovers the market roles declared on ``schema`` (via ``With[...]``), attaches
    each one against ``snapshot``, projects to the schema's columns and validates.
    The pricing graph never sees ``snapshot`` — this is the whole market boundary.
    Stays lazy.
    """
    lf = raw.lazy() if isinstance(raw, pl.DataFrame) else raw
    for role in roles_of(schema):
        lf = role.attach(lf, snapshot)
    columns = list(schema.to_schema().columns.keys())
    return schema.validate(lf.select(columns), lazy=True)
