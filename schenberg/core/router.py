"""Router: a contract-oriented choice among pricing computations (ArrowChoice).

A router is not merely a list of filters. It is a *case split* over computations
that all satisfy the **same view contract**: every branch must produce the
declared ``view`` (e.g. ``OptionPrice``), so the choice is total over the
contract no matter which branch a row takes. Branches are ``FormulaGraph``\\ s (or
nested routers) — anything with the shared computation interface
(:class:`Computation`). The implementation still filters per branch and
``concat``\\ s, but the *semantics* are "choose among computations with one
contract", and the output is normalized to that contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast, runtime_checkable

import polars as pl

from schenberg.core.columns import ColumnRef, RoutePredicate

if TYPE_CHECKING:
    from schenberg.market_data.snapshot import MarketSnapshot


@runtime_checkable
class Computation(Protocol):
    """The interface a router branch must satisfy — shared with FormulaGraph."""

    def compute(
        self,
        frame: pl.LazyFrame,
        *,
        market: MarketSnapshot | None = None,
        view: str = "result",
    ) -> pl.LazyFrame: ...

    def has_view(self, view: str) -> bool: ...

    def view_schema(self, view: str) -> object | None: ...


C = TypeVar("C", bound=Computation)

EXCLUSIVE = "exclusive"
FIRST_MATCH = "first_match"


@dataclass(slots=True)
class Branch:
    predicates: tuple[RoutePredicate | pl.Expr, ...]
    computation: Computation
    label: str
    key: tuple[object, ...] | None  # the case() values, for duplicate detection


@dataclass(slots=True)
class Router:
    """Dispatch rows to per-case computations under one view contract.

    Build with :meth:`on` over the route columns, declare the shared contract with
    :meth:`returns`, then register cases with :meth:`case` (equality on the route
    columns) or :meth:`when` (arbitrary predicates). Unmatched rows fall to
    :meth:`default`. The default mode is :meth:`exclusive`; :meth:`first_match`
    opts into priority order.
    """

    route_columns: tuple[ColumnRef, ...]
    branches: list[Branch] = field(default_factory=list)
    fallback: Computation | None = None
    mode: str = EXCLUSIVE
    contract_view: str | None = None
    contract_schema: object | None = None

    @classmethod
    def on(cls, *columns: ColumnRef) -> Router:
        if not columns:
            raise ValueError("Router.on(...) requires at least one route column")
        return cls(route_columns=tuple(columns))

    # ---- contract & mode -------------------------------------------------

    def returns(self, view: str, schema: object | None = None) -> Router:
        """Declare the view contract every branch must satisfy."""
        self.contract_view = view
        self.contract_schema = schema
        return self

    def exclusive(self) -> Router:
        self.mode = EXCLUSIVE
        return self

    def first_match(self) -> Router:
        self.mode = FIRST_MATCH
        return self

    # ---- registration ----------------------------------------------------

    def default(self, computation: C) -> Router:
        self._check_contract(computation, "default")
        self.fallback = computation
        return self

    def case(self, *values: object) -> Callable[[Callable[[], C]], C]:
        """Register a case by value, one per route column (equality predicates).

        In :attr:`EXCLUSIVE` mode a duplicate case key is rejected at registration.
        """
        if len(values) != len(self.route_columns):
            raise ValueError(
                f"case expects {len(self.route_columns)} value(s) for route columns "
                f"{[c.name for c in self.route_columns]}, got {len(values)}"
            )
        # Unwrap enum members to their underlying value so the predicate compares
        # against the (string/int) route column, while accepting enums in the API.
        unwrapped = tuple(getattr(v, "value", v) for v in values)
        predicates = tuple(
            column == value for column, value in zip(self.route_columns, unwrapped, strict=True)
        )
        key = unwrapped
        label = ", ".join(str(v) for v in unwrapped)
        return self._register(predicates, key=key, label=label)

    def when(self, *predicates: RoutePredicate | pl.Expr) -> Callable[[Callable[[], C]], C]:
        """Register a case by explicit predicates (supports complex conditions)."""
        label = " & ".join(_predicate_label(p) for p in predicates)
        return self._register(predicates, key=None, label=label)

    def _register(
        self,
        predicates: tuple[RoutePredicate | pl.Expr, ...],
        *,
        key: tuple[object, ...] | None,
        label: str,
    ) -> Callable[[Callable[[], C]], C]:
        if self.mode == EXCLUSIVE and key is not None:
            for existing in self.branches:
                if existing.key == key:
                    raise ValueError(
                        f"duplicate case {key} in exclusive router; "
                        f"use first_match() for priority ordering"
                    )

        def decorator(builder: Callable[[], C]) -> C:
            computation = builder()
            self._check_contract(computation, label)
            self.branches.append(Branch(tuple(predicates), computation, label, key))
            return computation

        return decorator

    def _check_contract(self, computation: Computation, where: str) -> None:
        if self.contract_view is None:
            return
        view = self.contract_view
        if not computation.has_view(view):
            raise ValueError(f"branch {where!r} does not provide the router view {view!r}")
        if self.contract_schema is not None:
            branch_schema = computation.view_schema(view)
            if branch_schema is not None and not _schema_compatible(
                branch_schema, self.contract_schema
            ):
                raise ValueError(
                    f"branch {where!r} view {view!r} schema "
                    f"{getattr(branch_schema, '__name__', branch_schema)!r} is not "
                    f"compatible with router contract "
                    f"{getattr(self.contract_schema, '__name__', self.contract_schema)!r}"
                )

    # ---- computation interface (a router is itself a Computation) --------

    def has_view(self, view: str) -> bool:
        return self.contract_view == view

    def view_schema(self, view: str) -> object | None:
        return self.contract_schema if self.contract_view == view else None

    def compute(
        self,
        frame: pl.LazyFrame,
        *,
        market: MarketSnapshot | None = None,
        view: str = "result",
    ) -> pl.LazyFrame:
        """Route rows to their branch and concat the results under the contract."""
        parts: list[pl.LazyFrame] = []
        matched = pl.lit(False)

        for branch in self.branches:
            condition = self._and(branch.predicates)
            selector = condition & ~matched if self.mode == FIRST_MATCH else condition
            parts.append(
                branch.computation.compute(frame.filter(selector), market=market, view=view)
            )
            matched = matched | condition

        if self.fallback is not None:
            parts.append(self.fallback.compute(frame.filter(~matched), market=market, view=view))

        if not parts:
            raise ValueError("router has no registered cases and no fallback")

        result = pl.concat(parts, how="diagonal_relaxed")
        return self._enforce_contract(result, view)

    def _enforce_contract(self, lf: pl.LazyFrame, view: str) -> pl.LazyFrame:
        """Normalize the concatenated output to the contract's columns, so a
        ``diagonal_relaxed`` concat can never silently widen the result."""
        if self.contract_view != view or self.contract_schema is None:
            return lf
        fields = list(cast(Any, self.contract_schema).to_schema().columns.keys())
        return lf.select(fields)

    # ---- introspection ---------------------------------------------------

    def info(self, *, view: str | None = None) -> dict[str, object]:
        return {
            "route_columns": [c.name for c in self.route_columns],
            "mode": self.mode,
            "view": self.contract_view,
            "schema": getattr(self.contract_schema, "__name__", self.contract_schema),
            "cases": [b.label for b in self.branches],
        }

    def explain(self, *, view: str | None = None) -> str:
        schema_name = getattr(self.contract_schema, "__name__", None)
        lines = [f"Router: {' , '.join(c.name for c in self.route_columns)}"]
        if schema_name:
            lines[0] += f" -> {schema_name}"
        lines += ["", "Mode:", f"  - {self.mode}", "", "Route terms:"]
        lines += [f"  - {c.name}" for c in self.route_columns]
        lines += ["", "Cases:"]
        lines += [f"  - {b.label}" for b in self.branches]
        if self.fallback is not None:
            lines.append("  - <default>")
        if self.contract_view is not None:
            lines += ["", "All cases return:", f"  - view: {self.contract_view}"]
            if schema_name:
                lines.append(f"  - schema: {schema_name}")
        lines += ["", "This is ArrowChoice in Schenberg."]
        return "\n".join(lines)

    def diagnose(self, frame: pl.LazyFrame, *, view: str = "result") -> pl.DataFrame:
        """Per-branch match counts plus the unmatched remainder — a cheap check
        that an exclusive router's cases truly partition the rows."""
        rows: list[dict[str, object]] = []
        matched = pl.lit(False)
        for branch in self.branches:
            condition = self._and(branch.predicates)
            count = cast(pl.DataFrame, frame.filter(condition).select(pl.len()).collect()).item()
            overlap = cast(
                pl.DataFrame, frame.filter(condition & matched).select(pl.len()).collect()
            ).item()
            rows.append({"case": branch.label, "matched": count, "overlap": overlap})
            matched = matched | condition
        unmatched = cast(pl.DataFrame, frame.filter(~matched).select(pl.len()).collect()).item()
        rows.append({"case": "<unmatched>", "matched": unmatched, "overlap": 0})
        return pl.DataFrame(rows)

    @staticmethod
    def _and(predicates: tuple[RoutePredicate | pl.Expr, ...]) -> pl.Expr:
        condition = pl.lit(True)
        for predicate in predicates:
            expr = predicate.expr() if isinstance(predicate, RoutePredicate) else predicate
            condition = condition & expr
        return condition


def _predicate_label(p: RoutePredicate | pl.Expr) -> str:
    if isinstance(p, RoutePredicate):
        return f"{p.column.name} {p.op} {getattr(p.value, 'value', p.value)}"
    return str(p)


def _schema_compatible(branch_schema: object, contract_schema: object) -> bool:
    """A branch satisfies the contract when its view fields cover the contract's."""
    if branch_schema is contract_schema:
        return True
    try:
        branch_fields = set(cast(Any, branch_schema).to_schema().columns.keys())
        contract_fields = set(cast(Any, contract_schema).to_schema().columns.keys())
    except AttributeError:
        return False
    return contract_fields <= branch_fields
