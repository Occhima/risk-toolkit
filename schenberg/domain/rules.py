from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl

RuleExpr = Callable[[type[Any]], pl.Expr]

_RULE_SPEC_ATTR = "__schenberg_rule_spec__"


@dataclass(frozen=True, slots=True)
class _RuleSpec:
    output: str
    selector: str
    value: object | None
    is_default: bool


@dataclass(frozen=True, slots=True)
class RuleCase:
    value: object | None
    expr: pl.Expr
    is_default: bool = False


@dataclass(frozen=True, slots=True)
class ContractRule:
    output: str
    selector: str
    cases: tuple[RuleCase, ...]

    def apply(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        names = set(lf.collect_schema().names())

        non_default = [c for c in self.cases if not c.is_default]
        defaults = [c for c in self.cases if c.is_default]

        if not non_default and not defaults:
            return lf

        selector_col = pl.col(self.selector)

        if non_default:
            first = non_default[0]
            val = getattr(first.value, "value", first.value)
            chain = pl.when(selector_col == val).then(first.expr)
            for case in non_default[1:]:
                cval = getattr(case.value, "value", case.value)
                chain = chain.when(selector_col == cval).then(case.expr)

            computed = chain.otherwise(defaults[0].expr) if defaults else chain.otherwise(None)
        else:
            computed = defaults[0].expr

        if self.output in names:
            return lf.with_columns(pl.coalesce([pl.col(self.output), computed]).alias(self.output))
        return lf.with_columns(computed.alias(self.output))


def rule_for(
    output: str,
    *,
    selector: str,
    value: object | None = None,
    default: bool = False,
) -> Callable[[RuleExpr], RuleExpr]:
    """Attach a contract-rule case to a method.

    The decorated method receives ``cls`` and returns a ``pl.Expr`` for the
    derived column (no alias needed — ``ContractRule.apply`` adds it).
    """

    def decorator(fn: RuleExpr) -> RuleExpr:
        spec = _RuleSpec(output=output, selector=selector, value=value, is_default=default)
        setattr(fn, _RULE_SPEC_ATTR, spec)
        return fn

    return decorator


def collect_rules_from_mro(cls: type[Any]) -> tuple[ContractRule, ...]:  # noqa: PLR0912
    """Walk the MRO and compile one ``ContractRule`` per ``(output, selector)`` pair.

    More-specific classes override base-class cases for the same
    ``(output, selector, normalized_value)`` triple.  Raises ``ValueError`` if
    the same ``output`` is declared with two different selectors (ambiguous write).
    """
    # (output, selector, norm_val) -> (spec, fn)
    # Walking reversed MRO (base first) so child writes win.
    override_map: dict[tuple[str, str, object], tuple[_RuleSpec, RuleExpr]] = {}

    for klass in reversed(cls.__mro__):
        for attr_val in vars(klass).values():
            if not callable(attr_val):
                continue
            spec: _RuleSpec | None = getattr(attr_val, _RULE_SPEC_ATTR, None)
            if spec is None:
                continue
            norm_val: object
            if spec.is_default:
                norm_val = "__default__"
            else:
                norm_val = getattr(spec.value, "value", spec.value)
            override_map[(spec.output, spec.selector, norm_val)] = (spec, attr_val)

    # Validate: each output must have at most one selector.
    output_selectors: dict[str, str] = {}
    for output, selector, _ in override_map:
        existing = output_selectors.get(output)
        if existing is not None and existing != selector:
            raise ValueError(
                f"output {output!r} declared with selectors {existing!r} and "
                f"{selector!r}; ambiguous in v1 — use one selector per output"
            )
        output_selectors[output] = selector

    # Group by (output, selector).
    groups: dict[tuple[str, str], list[tuple[_RuleSpec, RuleExpr]]] = {}
    for (output, selector, _), pair in override_map.items():
        groups.setdefault((output, selector), []).append(pair)

    # Compile ContractRule objects (evaluate each fn with cls once).
    rules: list[ContractRule] = []
    for (output, selector), pairs in groups.items():
        cases: list[RuleCase] = []
        for spec, fn in pairs:
            cases.append(RuleCase(value=spec.value, expr=fn(cls), is_default=spec.is_default))
        rules.append(ContractRule(output=output, selector=selector, cases=tuple(cases)))

    # Merge in any explicit __rules__ declared directly on a class (not inherited).
    covered = {(r.output, r.selector) for r in rules}
    for klass in reversed(cls.__mro__):
        if "__rules__" in vars(klass):
            for rule in vars(klass)["__rules__"]:
                if (rule.output, rule.selector) not in covered:
                    rules.append(rule)
                    covered.add((rule.output, rule.selector))

    return tuple(rules)
