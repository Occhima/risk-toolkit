from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd
import pandera.polars as pa
import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.domain.rules import ContractRule, collect_rules_from_mro


class ContractAdapterMixin:
    """Convenience constructors for schema-bound LazyFrames."""

    @classmethod
    def from_pandas(cls: type[Any], data: pd.DataFrame) -> LazyFrame[Any]:
        lf = pl.DataFrame(data.to_dict(orient="list")).lazy()
        return cls.validate(lf, lazy=True)

    @classmethod
    def from_polars(cls: type[Any], data: pl.DataFrame | pl.LazyFrame) -> LazyFrame[Any]:
        lf = data.lazy() if isinstance(data, pl.DataFrame) else data
        return cls.validate(lf, lazy=True)

    @classmethod
    def from_records(cls: type[Any], records: list[dict[str, Any]]) -> LazyFrame[Any]:
        return cls.from_polars(pl.DataFrame(records))

    @classmethod
    def from_vectors(cls: type[Any], **vectors: Any) -> LazyFrame[Any]:
        return cls.from_polars(pl.DataFrame(vectors))


class SchenbergDataFrameModel(ContractAdapterMixin, pa.DataFrameModel):
    """Base Pandera model for all Schenberg dataframe contracts.

    Contract rules declared with ``@rule_for`` are applied in ``validate()``
    before Pandera schema checks run.  Callers use ``pa.check_types(lazy=True)``
    on their pricing functions; no manual ``resolve()`` call is needed.
    """

    __rules__: ClassVar[tuple[ContractRule, ...]] = ()

    @classmethod
    def resolve(cls, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply all contract rules from the MRO, filling derived coordinates."""
        for rule in collect_rules_from_mro(cls):
            lf = rule.apply(lf)
        return lf

    @classmethod
    def to_schema(cls):
        schema = super().to_schema()
        original_validate = schema.validate

        def _resolve_then_validate(
            check_obj: pl.LazyFrame,
            head: int | None = None,
            tail: int | None = None,
            sample: int | None = None,
            random_state: int | None = None,
            lazy: bool = False,
            inplace: bool = False,
        ) -> pl.LazyFrame:
            return original_validate(cls.resolve(check_obj), head, tail, sample, random_state, lazy, inplace)

        schema.validate = _resolve_then_validate
        return schema

    @classmethod
    def validate(cls, check_obj: pl.LazyFrame, *args: Any, **kwargs: Any) -> pl.LazyFrame:
        return super().validate(cls.resolve(check_obj), *args, **kwargs)

    class Config:
        coerce = True
        strict = False
