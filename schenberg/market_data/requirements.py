"""Declarative market requirements for contract-oriented pricing graphs.

A pricing graph is *pure*: its formulas read contract columns and market columns
as :class:`~schenberg.core.graph.Term`\\ s and never join anything themselves. The
joins live here, declared once, as a typed schema of *what market data this
instrument needs and how to find each row*::

    class EnergyForwardRequirements(MarketRequirements[EnergyForwardContract]):
        zero_rate: Term[float] = requires(
            CURVES.zero_rate().by(curve=contract.discount_curve, tenor=contract.payment_date)
        )

Three ideas make this read well:

* **The field name is the output column.** ``zero_rate: Term[float]`` says "this
  instrument exposes a ``zero_rate`` market column"; there is no second place to
  name the output, and the annotation doubles as the reference type.
* **``requires(...)`` wraps a fluent read.** ``CURVES.zero_rate()`` selects the
  table and value; ``.by(key=contract.col)`` supplies the join keys. ``.by`` is
  *optional* — a read carries typed default key columns, so you write it only for
  the rare contract that names its columns differently.
* **``contract.x`` is a validated proxy.** Every reference is checked against the
  ``[EnergyForwardContract]`` schema when the class is created, so a typo fails at
  import, not at ``collect()``.

At class creation each field is resolved to the engine's existing
:class:`~schenberg.core.market.MarketRequirement` (output = field name) and the
attribute is rebound to the resulting MARKET ``Term`` so formulas can simply
``uses(m.zero_rate)``. The compiled dependencies are the *same* objects
``FormulaGraph.market`` already consumes — this layer is a nicer face on a proven
backend, not a new join engine.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, ClassVar, get_args

from schenberg.core.columns import ColumnBinding, ColumnSet
from schenberg.core.graph import Term, TermKind
from schenberg.core.market import MarketDependency, MarketRequirement


@dataclass(frozen=True, slots=True)
class ContractRef:
    """A symbolic reference to one contract column: ``contract.discount_curve``."""

    name: str


class _ContractProxy:
    """Attribute access yields a :class:`ContractRef`. The single module-level
    ``contract`` instance is untyped on its own; references are validated against
    the concrete contract schema when a :class:`MarketRequirements` subclass is
    built (see :meth:`MarketRequirements.__init_subclass__`)."""

    def __getattr__(self, name: str) -> ContractRef:
        if name.startswith("_"):
            raise AttributeError(name)
        return ContractRef(name)


#: Symbolic stand-in for "a column of this instrument's contract". Use it inside
#: ``.by(...)`` to point a join key at a contract column: ``tenor=contract.payment_date``.
contract = _ContractProxy()


def _ref_name(ref: ContractRef | str | Any) -> str:
    if isinstance(ref, ContractRef):
        return ref.name
    if isinstance(ref, str):
        return ref
    name = getattr(ref, "name", None)  # tolerate a graph Term / ColumnRef
    if isinstance(name, str):
        return name
    raise TypeError(f"cannot use {ref!r} as a contract column reference")


@dataclass(frozen=True, slots=True)
class Key:
    """One join key of a market read: the semantic name used in ``.by()``, the
    quote-side column it matches, and the contract column it defaults to."""

    name: str  # the keyword accepted by .by(...)  e.g. "tenor"
    quote_col: str  # the source/quote-side join column  e.g. "tenor_days"
    default: str  # the contract column used when .by() omits this key


@dataclass(frozen=True, slots=True)
class Keyed:
    """A fluent, not-yet-bound keyed read of a market table.

    A market spec hands one back (``CURVES.zero_rate()``); ``.by(...)`` overrides
    individual join keys; :meth:`finalize` resolves it into the engine's
    :class:`MarketRequirement` once the output column (the requirements field name)
    is known. Immutable — ``.by`` returns a new ``Keyed``.
    """

    table: str
    value_col: str
    keys: tuple[Key, ...]
    overrides: tuple[tuple[str, str], ...] = ()  # (key name -> contract column)

    def by(self, **bindings: ContractRef | str) -> Keyed:
        """Point one or more join keys at specific contract columns.

        ``.by(curve=contract.discount_curve, tenor=contract.payment_date)``. Keys
        left out keep their typed defaults; an unknown key name is rejected here,
        not silently ignored.
        """
        known = {k.name for k in self.keys}
        unknown = sorted(set(bindings) - known)
        if unknown:
            raise ValueError(
                f"{self.table}.{self.value_col}: unknown join key(s) {unknown}; "
                f"this read keys on {sorted(known)}"
            )
        merged = dict(self.overrides)
        merged.update({name: _ref_name(ref) for name, ref in bindings.items()})
        return replace(self, overrides=tuple(merged.items()))

    def finalize(
        self,
        output: str,
        *,
        schema_columns: set[str] | None = None,
        where: str = "",
    ) -> MarketRequirement:
        over = dict(self.overrides)
        bindings: list[ColumnBinding] = []
        for key in self.keys:
            left = over.get(key.name, key.default)
            if schema_columns is not None and left not in schema_columns:
                raise ValueError(
                    f"{where}: join key {key.name!r} maps to contract column {left!r}, "
                    f"which is not a column of the contract schema"
                )
            bindings.append(ColumnBinding(left, key.quote_col))
        return MarketRequirement(
            table=self.table,
            on=ColumnSet(tuple(bindings)),
            outputs={self.value_col: output},
        )


@dataclass(frozen=True, slots=True)
class Requirement:
    """The value a class field holds before resolution: a wrapped fluent read."""

    read: Keyed


def requires(read: Keyed) -> Any:
    """Declare a market requirement field. Returns ``Any`` so the field can carry a
    clean ``Term[float]`` annotation while holding the read until the class is
    built."""
    return Requirement(read)


def _contract_arg(cls: type) -> type | None:
    """The concrete contract type from ``MarketRequirements[SomeContract]``."""
    for base in getattr(cls, "__orig_bases__", ()):
        args = get_args(base)
        if args:
            return args[0]
    return None


class MarketRequirements[C]:
    """Base class for an instrument's market-data schema. Subclass it with the
    contract type and declare one ``requires(...)`` field per market column.

    The resolved, attachable dependencies are exposed as ``__requirements__`` (a
    ``{field_name: MarketDependency}`` map) for a :class:`PricingGraph` to attach;
    each field attribute is rebound to its MARKET ``Term`` for use in formulas.
    """

    __requirements__: ClassVar[dict[str, MarketDependency]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        contract_type = _contract_arg(cls)
        schema_columns: set[str] | None = None
        if contract_type is not None and hasattr(contract_type, "to_schema"):
            schema_columns = set(contract_type.to_schema().columns.keys())

        resolved: dict[str, MarketDependency] = {}
        for name, value in list(vars(cls).items()):
            if isinstance(value, Requirement):
                resolved[name] = value.read.finalize(
                    name, schema_columns=schema_columns, where=f"{cls.__name__}.{name}"
                )
                setattr(cls, name, Term(name=name, kind=TermKind.MARKET))
        cls.__requirements__ = resolved
