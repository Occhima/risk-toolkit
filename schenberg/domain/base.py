from __future__ import annotations

from typing import Any

import pandas as pd
import pandera.polars as pa
import polars as pl
from pandera.typing.polars import LazyFrame


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


class DataFrameModel(ContractAdapterMixin, pa.DataFrameModel):
    """Base Pandera model for all Schenberg dataframe contracts."""

    class Config:
        coerce = True
        strict = False
