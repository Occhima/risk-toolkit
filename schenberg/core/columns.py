"""Small helpers for schema-derived column references and join bindings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass(frozen=True, slots=True)
class ColumnRef:
    name: str

    def expr(self) -> pl.Expr:
        return pl.col(self.name)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> RoutePredicate:  # type: ignore[override]  # ty: ignore[invalid-method-override]
        return RoutePredicate(self, "==", other)


@dataclass(frozen=True, slots=True)
class RoutePredicate:
    column: ColumnRef
    op: str
    value: object

    def expr(self) -> pl.Expr:
        if self.op == "==":
            return pl.col(self.column.name).eq(self.value)
        raise ValueError(f"unsupported route operation: {self.op!r}")


class SchemaColumns:
    def __init__(self, schema: type[Any]) -> None:
        self._schema = schema
        self._names = set(schema.to_schema().columns.keys())

    def __getattr__(self, name: str) -> ColumnRef:
        if name not in self._names:
            raise AttributeError(f"{name!r} is not a declared column in {self._schema.__name__}")
        return ColumnRef(name)


def cols(schema: type[Any]) -> SchemaColumns:
    return SchemaColumns(schema)


@dataclass(frozen=True, slots=True)
class ColumnBinding:
    left: str
    right: str


@dataclass(frozen=True, slots=True)
class ColumnSet:
    bindings: tuple[ColumnBinding, ...]

    @classmethod
    def from_pairs(cls, *pairs: tuple[str, str]) -> ColumnSet:
        return cls(tuple(ColumnBinding(left, right) for left, right in pairs))

    @property
    def left_keys(self) -> tuple[str, ...]:
        return tuple(binding.left for binding in self.bindings)

    @property
    def right_keys(self) -> tuple[str, ...]:
        return tuple(binding.right for binding in self.bindings)
