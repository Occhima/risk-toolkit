You are a local coding model with full access to the Schenberg Risk Toolkit repository.

Implement the feature below carefully. Make small, local, testable changes. Do not rewrite the whole project.

The user has deleted `schenberg/pricing/instruments` locally, so do not depend on that folder for this task. Focus on:

* `schenberg/domain`
* `schenberg/core`
* `schenberg/market_data`
* docs
* focused tests

This is an intentional breaking change. Do not preserve old APIs.

# Goal

Implement two things:

2. Use `dataclasses` only where they make declarations/specs cleaner and safer.
3. Implement contract-rule injection through `SchenbergDataFrameModel.validate()`.

The final usage must look like this:

```python
class IndexerFixingMixin(SchenbergDataFrameModel):
    indexer: IndexerEnum
    index_fixing_date: date | None = None

    @rule_for("index_fixing_date", selector="indexer", value=IndexerEnum.CPI)
    def _cpi(cls):
        return dates.add_days("tenor", 5)

    @rule_for("index_fixing_date", selector="indexer", default=True)
    def _default(cls):
        return dates.same_day("tenor")


class CurrencyFixingMixin(SchenbergDataFrameModel):
    currency: CurrencyEnum
    currency_fixing_date: date | None = None

    @rule_for("currency_fixing_date", selector="currency", value=CurrencyEnum.EUR)
    def _eur(cls):
        return dates.previous_business_day("tenor")

    @rule_for("currency_fixing_date", selector="currency", default=True)
    def _default(cls):
        return dates.same_day("tenor")


class ForwardContractPricing(
    IndexerFixingMixin,
    CurrencyFixingMixin,
    TenorMixin,
    SchenbergDataFrameModel,
):
    tenor: date
    indexer: IndexerEnum
    currency: CurrencyEnum
    strike: float
    payment_days: int


class EnergyForwardPricing(ForwardContractPricing):
    submarket: SubmarketEnum
    incentive: IncentiveEnum

    @rule_for("index_fixing_date", selector="indexer", value=IndexerEnum.PLD)
    def _pld(cls):
        return dates.nth_business_day_next_month("tenor", n=6)


@pa.check_types(lazy=True)
def price_energy_forward(
    trades: LazyFrame[EnergyForwardPricing],
    market: MarketSnapshot,
) -> LazyFrame[ForwardPricing]:
    return forward_formula.compute(trades, market=market, view="output")
```

No extra `resolve_contract` decorator. No manual `resolve()` call in pricing functions. The contract must resolve itself through `SchenbergDataFrameModel.validate()`.

# Current files to inspect first

Before editing, read:

* `schenberg/domain/base.py`
* `schenberg/core/graph.py`
* `schenberg/core/__init__.py`
* `schenberg/market_data/date_rules.py`
* `schenberg/market_data/requirements.py`
* `schenberg/pricing/market.py`
* `docs/concepts.md`
* `docs/extending.md`
* `pyproject.toml`

The current base model is probably called `DataFrameModel`. Replace it with `SchenbergDataFrameModel`.

This is a breaking change. Update all internal imports/usages accordingly. Do not keep `DataFrameModel` as an alias.

# Core design rules

Follow these strictly:

* `SchenbergDataFrameModel` is the base class for Schenberg dataframe contracts.
* Contract rules are declared on schemas and mixins.
* `SchenbergDataFrameModel.validate()` resolves contract rules before calling `super().validate(...)`.
* `pa.check_types(lazy=True)` must continue to be used directly on pricing functions.
* The graph must never compute fixing dates.
* The market snapshot must never compute fixing dates.
* No `.collect()` in runtime code.
* `collect_schema()` is allowed.
* Do not use `map_elements`.
* Do not use global registries.
* Do not use import side effects.
* Do not use `Router` for fixing-date conventions.
* Do not depend on `schenberg/pricing/instruments`.

Conceptual rule:

```text
fixing_value is market data.
index_fixing_date, currency_fixing_date, base_index_date, projection_date, etc. are contractual coordinates.
```

The contract provides coordinates. The market provides values.


# Part 2 — Use dataclasses where useful

Use `dataclasses` only where they simplify the declaration/spec surface.

Good candidates:

* `Formula` if it is a thin public facade.
* Contract rule specs.
* Compiled rule groups.
* Date-rule specs.

Do not force the low-level mutable graph engine into a dataclass if that creates risk.

Do not make stateful graph internals frozen.

The goal is cleaner declarations, not a full engine rewrite.

# Part 3 — Create `SchenbergDataFrameModel`

Edit:

```text
schenberg/domain/base.py
```

Replace the current Schenberg dataframe base with:

```python
class SchenbergDataFrameModel(ContractAdapterMixin, pa.DataFrameModel):
    __rules__: ClassVar[tuple[ContractRule, ...]] = ()

    @classmethod
    def resolve(cls, lf: pl.LazyFrame) -> pl.LazyFrame:
        for rule in collect_rules_from_mro(cls):
            lf = rule.apply(lf)
        return lf

    @classmethod
    def validate(cls, check_obj, *args, **kwargs):
        if isinstance(check_obj, pl.DataFrame):
            check_obj = check_obj.lazy()

        if isinstance(check_obj, pl.LazyFrame):
            check_obj = cls.resolve(check_obj)

        return super().validate(check_obj, *args, **kwargs)

    class Config:
        coerce = True
        strict = False
```

Preserve the existing convenience constructors, but update them to use `SchenbergDataFrameModel`.

Important:

* Do not keep `DataFrameModel` as an alias.
* Update all repository imports from `DataFrameModel` to `SchenbergDataFrameModel`.
* `resolve()` must not call `validate()`.
* `validate()` must call `resolve()` and then `super().validate(...)`.
* Avoid recursion.
* Do not catch or hide Pandera errors.
* Do not call `.collect()`.

# Part 4 — Create contract rules module

Create:

```text
schenberg/domain/rules.py
```

Implement the rule system used by `SchenbergDataFrameModel`.

Required imports:

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl
```

Implement two levels:

1. Raw rule specs attached by the decorator.
2. Compiled `ContractRule` objects returned by `collect_rules_from_mro(cls)`.

The final `SchenbergDataFrameModel.resolve()` must work exactly like:

```python
for rule in collect_rules_from_mro(cls):
    lf = rule.apply(lf)
```

So `ContractRule` must have an `.apply(lf)` method.

Suggested structure:

```python
RuleExpr = Callable[[type[Any]], pl.Expr]


@dataclass(frozen=True, slots=True)
class RuleCase:
    value: object | None
    expr: RuleExpr
    is_default: bool = False


@dataclass(frozen=True, slots=True)
class ContractRule:
    output: str
    selector: str
    cases: tuple[RuleCase, ...]

    def apply(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        ...
```

The decorator:

```python
def rule_for(
    output: str,
    *,
    selector: str,
    value: object | None = None,
    default: bool = False,
) -> Callable[[RuleExpr], RuleExpr]:
    ...
```

Behavior:

* Attach metadata to the decorated function.
* Example attribute name: `__schenberg_rule_spec__`.
* The decorated function receives `cls` and returns `pl.Expr`.
* The returned `pl.Expr` does not need an alias.
* The compiled `ContractRule.apply()` must alias to `output`.

Expected usage:

```python
@rule_for("index_fixing_date", selector="indexer", value=IndexerEnum.CPI)
def _cpi(cls):
    return dates.add_days("tenor", 5)

@rule_for("index_fixing_date", selector="indexer", default=True)
def _default(cls):
    return dates.same_day("tenor")
```

# Part 5 — MRO rule collection

Implement:

```python
def collect_rules_from_mro(cls: type[Any]) -> tuple[ContractRule, ...]:
    ...
```

Requirements:

* Walk Python MRO.
* Mixins compose.
* More specific classes override base classes.
* No global registry.
* No import side effects.
* Support direct explicit `__rules__` on classes if present.
* Support decorated methods via `@rule_for`.

Override identity:

For non-default cases:

```python
(output, selector, normalized_value)
```

For default cases:

```python
(output, selector, "__default__")
```

Normalize enum values with:

```python
getattr(value, "value", value)
```

If a child class redeclares the same `(output, selector, value)`, it must replace the parent rule.

Example:

```python
class BaseIndexerFixingMixin(SchenbergDataFrameModel):
    indexer: IndexerEnum
    index_fixing_date: date | None = None

    @rule_for("index_fixing_date", selector="indexer", value=IndexerEnum.CPI)
    def _cpi(cls):
        return dates.add_days("tenor", 5)


class EnergyForwardPricing(BaseIndexerFixingMixin):
    @rule_for("index_fixing_date", selector="indexer", value=IndexerEnum.CPI)
    def _cpi_energy(cls):
        return dates.add_days("tenor", 10)
```

`EnergyForwardPricing` must use `+10`, not `+5`.

Group compiled rules by `(output, selector)`.

If the same `output` is declared with different selectors, either support it safely or raise a clear `ValueError`. Prefer raising in v1 to avoid ambiguous writes.

# Part 6 — ContractRule.apply semantics

`ContractRule.apply(lf)` must:

1. Build a lazy expression for one output column.
2. Use the selector column to choose cases.
3. Preserve user-provided non-null values.
4. Fill null/missing values.
5. Never collect.

Pseudo-behavior:

```python
names = set(lf.collect_schema().names())

computed = (
    pl.when(pl.col(selector) == case_value_1).then(case_expr_1)
    .when(pl.col(selector) == case_value_2).then(case_expr_2)
    ...
    .otherwise(default_expr_or_None)
    .alias(output)
)

if output in names:
    return lf.with_columns(
        pl.coalesce([pl.col(output), computed]).alias(output)
    )

return lf.with_columns(computed)
```

Important:

* Normalize enum values before comparing.
* Default case uses `.otherwise(default_expr)`.
* If no default exists, use `.otherwise(None)`.
* If user passed `index_fixing_date`, keep it.
* If user passed null, fill it.
* If the column is missing, create it.
* Do not call `.collect()`.

# Part 7 — Date helpers

Edit:

```text
schenberg/market_data/date_rules.py
```

Keep existing helpers. Add minimal helpers if missing:

```python
def same_day(anchor: str, *, output_col: str | None = None) -> pl.Expr:
    expr = pl.col(anchor)
    return expr if output_col is None else expr.alias(output_col)
```

```python
def add_days(anchor: str, days: int, *, output_col: str | None = None) -> pl.Expr:
    expr = pl.col(anchor) + pl.duration(days=days)
    return expr if output_col is None else expr.alias(output_col)
```

```python
def previous_day(anchor: str, *, output_col: str | None = None) -> pl.Expr:
    expr = pl.col(anchor) - pl.duration(days=1)
    return expr if output_col is None else expr.alias(output_col)
```

If `previous_business_day` or `nth_business_day_next_month` already exists or is trivial to build from existing code, expose them. Otherwise do not implement full calendar complexity in this task.

Do not use `map_elements`.

# Part 8 — Tests for rules

Create:

```text
tests/domain/test_contract_rules.py
```

Do not depend on `schenberg/pricing/instruments`.

Use dummy schemas.

Imports:

```python
from __future__ import annotations

from datetime import date
from enum import StrEnum

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.domain.base import SchenbergDataFrameModel
from schenberg.domain.rules import rule_for
from schenberg.market_data import date_rules as dates
```

Dummy enum:

```python
class IndexerEnum(StrEnum):
    CPI = "CPI"
    IPCA = "IPCA"
    PLD = "PLD"
```

Schema:

```python
class ForwardContractPricing(SchenbergDataFrameModel):
    tenor: date
    indexer: IndexerEnum
    index_fixing_date: date

    @rule_for("index_fixing_date", selector="indexer", value=IndexerEnum.CPI)
    def _cpi(cls):
        return dates.add_days("tenor", 5)

    @rule_for("index_fixing_date", selector="indexer", default=True)
    def _default(cls):
        return dates.same_day("tenor")
```

Test 1:

* Define:

```python
@pa.check_types(lazy=True)
def identity_contract(
    df: LazyFrame[ForwardContractPricing],
) -> LazyFrame[ForwardContractPricing]:
    return df
```

* Input omits `index_fixing_date`.
* Function must return `pl.LazyFrame`.
* Collect only inside the test.
* CPI row gets `tenor + 5 days`.
* IPCA row gets `tenor`.

Test 2:

* Input includes `index_fixing_date` with a non-null explicit value.
* Explicit value must be preserved.

Test 3:

* Input includes `index_fixing_date` with null on one row.
* Null is filled.
* Non-null is preserved.

Test 4:

* Child class overrides CPI from `+5` to `+10`.
* Validate that child uses `+10`.

Test 5:

Create currency mixin:

```python
class CurrencyEnum(StrEnum):
    BRL = "BRL"
    EUR = "EUR"


class CurrencyFixingMixin(SchenbergDataFrameModel):
    currency: CurrencyEnum
    currency_fixing_date: date

    @rule_for("currency_fixing_date", selector="currency", value=CurrencyEnum.EUR)
    def _eur(cls):
        return dates.previous_day("tenor")

    @rule_for("currency_fixing_date", selector="currency", default=True)
    def _default(cls):
        return dates.same_day("tenor")
```

Final schema:

```python
class FullForwardContract(
    ForwardContractPricing,
    CurrencyFixingMixin,
    SchenbergDataFrameModel,
):
    currency: CurrencyEnum
    strike: float
```

Expected:

* `index_fixing_date` is filled.
* `currency_fixing_date` is filled.
* Both are validated by Pandera.
* Mixins do not conflict.

Test 6:

* Confirm new runtime files do not call `.collect()`.
* Grep or AST check:

  * `schenberg/domain/base.py`
  * `schenberg/domain/rules.py`
* Allow `collect_schema()`.

# Part 9 — Minimal Formula test

Create:

```text
tests/core/test_formula_contract_resolution.py
```

Do not depend on `schenberg/pricing/instruments`.

Use:

```python
from schenberg.core import Formula, uses
from schenberg.domain.base import SchenbergDataFrameModel
```

Define:

```python
class ForwardPricing(SchenbergDataFrameModel):
    value: float


class EnergyForwardPricing(ForwardContractPricing):
    submarket: str
    incentive: str
    strike: float
```

Create a tiny formula:

```python
formula = Formula("dummy_forward", input=EnergyForwardPricing)
c = formula.input

@formula.formula()
def value(strike: pl.Expr = uses(c.strike)) -> pl.Expr:
    return strike

formula.returns("output", ForwardPricing, value=value)
```

Pricing function:

```python
@pa.check_types(lazy=True)
def price_dummy_forward(
    trades: LazyFrame[EnergyForwardPricing],
) -> LazyFrame[ForwardPricing]:
    return formula.compute(trades, view="output")
```

Input omits `index_fixing_date`.

Expected:

* `pa.check_types` triggers `EnergyForwardPricing.validate()`.
* `validate()` resolves `index_fixing_date`.
* Function succeeds.
* Output is lazy.
* Collect only in test to assert `value`.

# Part 10 — Update docs

Update `docs/extending.md` with a concise section:

```text
Contract rules and derived contractual coordinates
```

Explain:

* Use `SchenbergDataFrameModel` as the base class.
* Mixins can declare rules using `@rule_for`.
* Rules derive contractual coordinates like `index_fixing_date`.
* User-provided values are preserved.
* Null/missing values are filled lazily.
* `pa.check_types(lazy=True)` remains the public validation mechanism.
* Formula graphs do not compute fixing dates.
* MarketSnapshot provides market values, not contractual coordinates.

Update docs/examples/tests to use `Formula`, not `PricingGraph`.

Do not add migration docs.

# Part 11 — Acceptance criteria

The implementation is complete when:

* `SchenbergDataFrameModel` exists and is the base contract model.
* `DataFrameModel` no longer exists as the public base class.
* All repo usages are updated to `SchenbergDataFrameModel`.
* `SchenbergDataFrameModel.validate()` resolves rules before Pandera validation.
* `SchenbergDataFrameModel.resolve()` uses:

```python
for rule in collect_rules_from_mro(cls):
    lf = rule.apply(lf)
```

* Mixins can inherit from `SchenbergDataFrameModel`.
* `@rule_for` works on mixins and child contracts.
* More specific classes override parent rules.
* Multiple mixins compose.
* User-provided non-null dates are preserved.
* Null dates are filled.
* Missing date columns are created.
* `pa.check_types(lazy=True)` works directly on pricing functions.
* No extra pricing decorator is needed.
* Public graph authoring API is `Formula`.
* `PricingGraph` does not exist.
* No alias `PricingGraph = Formula` exists.
* No compatibility shim exists.
* This command returns no real usages:

```bash
rg "PricingGraph" schenberg tests docs examples
```

* No runtime `.collect()` calls are introduced.
* Tests pass:

```bash
uv run pytest tests/domain/test_contract_rules.py
uv run pytest tests/core/test_formula_contract_resolution.py
uv run pytest
uv run ruff check .
uv run ty check
```

# Part 12 — What not to do

Do not:

* Keep `DataFrameModel` as alias.
* Keep `PricingGraph`.
* Add `PricingGraph = Formula`.
* Add migration shims.
* Add compatibility imports.
* Add migration docs.
* Add a `resolve_contract` decorator.
* Require users to call `resolve()` manually in pricing functions.
* Modify `MarketSnapshot`.
* Use `Router` for fixing-date conventions.
* Compute fixing dates inside a formula graph.
* Use `map_elements`.
* Call `.collect()` in runtime code.
* Depend on `schenberg/pricing/instruments`.
* Implement a huge calendar system.
* Rewrite all instruments.
* Rewrite the graph engine.
* Add global rule registries.

# Final shape

The final shape must look like this:

```python
class SchenbergDataFrameModel(pa.DataFrameModel):
    __rules__: ClassVar[tuple[ContractRule, ...]] = ()

    @classmethod
    def resolve(cls, lf: pl.LazyFrame) -> pl.LazyFrame:
        for rule in collect_rules_from_mro(cls):
            lf = rule.apply(lf)
        return lf

    @classmethod
    def validate(cls, check_obj, *args, **kwargs):
        if isinstance(check_obj, pl.DataFrame):
            check_obj = check_obj.lazy()

        if isinstance(check_obj, pl.LazyFrame):
            check_obj = cls.resolve(check_obj)

        return super().validate(check_obj, *args, **kwargs)


class IndexerFixingMixin(SchenbergDataFrameModel):
    indexer: IndexerEnum
    index_fixing_date: date | None = None

    @rule_for("index_fixing_date", selector="indexer", value=IndexerEnum.CPI)
    def _cpi(cls):
        return dates.add_days("tenor", 5)

    @rule_for("index_fixing_date", selector="indexer", default=True)
    def _default(cls):
        return dates.same_day("tenor")


class CurrencyFixingMixin(SchenbergDataFrameModel):
    currency: CurrencyEnum
    currency_fixing_date: date | None = None

    @rule_for("currency_fixing_date", selector="currency", value=CurrencyEnum.EUR)
    def _eur(cls):
        return dates.previous_business_day("tenor")

    @rule_for("currency_fixing_date", selector="currency", default=True)
    def _default(cls):
        return dates.same_day("tenor")


class ForwardContractPricing(
    IndexerFixingMixin,
    CurrencyFixingMixin,
    TenorMixin,
    SchenbergDataFrameModel,
):
    tenor: date
    indexer: IndexerEnum
    currency: CurrencyEnum
    strike: float
    payment_days: int


class EnergyForwardPricing(ForwardContractPricing):
    submarket: SubmarketEnum
    incentive: IncentiveEnum

    @rule_for("index_fixing_date", selector="indexer", value=IndexerEnum.PLD)
    def _pld(cls):
        return dates.nth_business_day_next_month("tenor", n=6)


@pa.check_types(lazy=True)
def price_energy_forward(
    trades: LazyFrame[EnergyForwardPricing],
    market: MarketSnapshot,
) -> LazyFrame[ForwardPricing]:
    return forward_formula.compute(trades, market=market, view="output")
```

The pricing function stays clean. The contract resolves itself. The graph remains pure.
