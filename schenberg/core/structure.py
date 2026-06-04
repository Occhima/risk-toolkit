"""Structure: component pricing -> exposure/weighting -> Fold.

A *structured instrument* is not priced by one formula. It is a composition:

1. price each component row purely (a :class:`~schenberg.core.graph.FormulaGraph`
   or :class:`~schenberg.core.router.Router` — no position direction);
2. apply **exposure** — the signed/scaled weight that turns a pure component value
   into this structure's contribution (``weighted_pv = pv * leg_weight``);
3. **fold** the contributions into one row per structure with a :class:`Fold`.

This is the home of everything a pure pricing graph must *not* know: pay/receive,
ativo/passivo, long/short, notional sign. A :class:`Structure` keeps those three
phases separate and inspectable — :meth:`components_frame` (pure prices),
:meth:`stage` (prices + exposure), :meth:`compute` (folded output) — plus
:meth:`explain`, :meth:`info` and :meth:`to_mermaid`. Nothing here calls
``.collect()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import polars as pl

from schenberg.core.columns import ColumnLike, SchemaColumns, col_name, cols
from schenberg.core.diagnostics import DiagnosticReport
from schenberg.core.fold import Agg, Fold
from schenberg.core.router import Computation

if TYPE_CHECKING:
    from schenberg.market_data.snapshot import MarketSnapshot


class Structure:
    """A structured instrument: pure component pricing, exposure, and a fold.

    Build it fluently::

        swap_structure = (
            Structure("swap", input=SwapLegInput)
            .components(swap_leg_router, view="pricing")
            .exposure(weighted_pv=pl.col("pv") * pl.col("leg_weight"))
            .fold(
                by="swap_id",
                returns=SwapOutput,
                npv=sum_("weighted_pv"),
                ativo_pv=sum_("weighted_pv", where=L.leg_role == "ativo"),
                passivo_pv=sum_("weighted_pv", where=L.leg_role == "passivo"),
            )
        )
    """

    def __init__(self, name: str, *, input: type[Any] | None = None) -> None:
        self.name = name
        self.input_schema = input
        self.component_computation: Computation | None = None
        self.component_view: str = "pricing"
        self.exposure_exprs: dict[str, pl.Expr] = {}
        self.fold_spec: Fold | None = None

    # ---- declaration -----------------------------------------------------

    @property
    def input(self) -> SchemaColumns:
        """Schema-derived column references for the structure's input rows."""
        if self.input_schema is None:
            raise AttributeError(f"structure {self.name!r} has no input schema")
        return cols(self.input_schema)

    def components(self, computation: Computation, *, view: str = "pricing") -> Structure:
        """Set the pure component pricing (a graph or router) and the view it
        publishes. The view must be pure component pricing — no exposure/sign."""
        self.component_computation = computation
        self.component_view = view
        return self

    def exposure(self, **exprs: pl.Expr) -> Structure:
        """Declare exposure/contribution columns added on top of component prices.

        This is where direction and weighting live — ``weighted_pv =
        pl.col("pv") * pl.col("leg_weight")`` — never inside the pricing graph.
        """
        self.exposure_exprs = dict(exprs)
        return self

    def fold(
        self,
        *,
        by: ColumnLike | list[ColumnLike] | tuple[ColumnLike, ...],
        returns: object | None = None,
        **aggregations: Agg | pl.Expr,
    ) -> Structure:
        """Declare the aggregation from component contributions to structure rows."""
        if isinstance(by, list | tuple):
            keys = [col_name(k) for k in cast("list[ColumnLike]", by)]
        else:
            keys = [col_name(by)]
        self.fold_spec = (
            Fold(f"{self.name}_fold", input_schema=self.input_schema)
            .by(*keys)
            .returns(returns, **aggregations)
        )
        return self

    # ---- interpretation (lazy) -------------------------------------------

    def components_frame(
        self, frame: pl.LazyFrame, *, market: MarketSnapshot | None = None
    ) -> pl.LazyFrame:
        """Pure component pricing: the priced component rows *before* exposure or
        fold. Carries the input columns alongside the pricing view. Stays lazy."""
        if self.component_computation is None:
            raise ValueError(f"structure {self.name!r} has no component computation")
        return self.component_computation.compute(
            frame, market=market, view=self.component_view
        )

    def stage(
        self, frame: pl.LazyFrame, *, market: MarketSnapshot | None = None
    ) -> pl.LazyFrame:
        """Debug interpretation: component prices *plus* exposure/contribution
        columns, one row per component, before aggregation. Stays lazy."""
        priced = self.components_frame(frame, market=market)
        if self.exposure_exprs:
            priced = priced.with_columns(
                **{name: expr for name, expr in self.exposure_exprs.items()}
            )
        return priced

    def compute(
        self,
        frame: pl.LazyFrame,
        *,
        market: MarketSnapshot | None = None,
        view: str | None = None,
    ) -> pl.LazyFrame:
        """Full interpretation: price components, apply exposure, fold to output.
        ``view`` is accepted for the shared computation interface and ignored
        (a structure has one output). Stays lazy."""
        if self.fold_spec is None:
            raise ValueError(f"structure {self.name!r} has no fold; call .fold(...)")
        return self.fold_spec.compute(self.stage(frame, market=market))

    # ---- validation ------------------------------------------------------

    def diagnose(self) -> DiagnosticReport:
        """Accumulate well-formedness issues into a :class:`DiagnosticReport`
        instead of raising on the first. Checks the structure has a component
        computation that provides its view, an exposure, and a fold whose group
        keys are set and whose output schema is covered."""
        report = DiagnosticReport()
        comp = self.component_computation
        is_router = hasattr(comp, "branches")
        if comp is None:
            report = report.add("error", "no-components", "no component computation set", self.name)
        elif not is_router and not comp.has_view(self.component_view):
            # A router passes input columns through and need not declare a contract
            # view; a plain graph must publish the component view it's asked for.
            report = report.add(
                "error",
                "missing-view",
                f"component computation does not provide view {self.component_view!r}",
                self.name,
            )
        if not self.exposure_exprs:
            report = report.add(
                "warning", "no-exposure", "structure has no exposure columns", self.name
            )
        if self.fold_spec is None:
            report = report.add("error", "no-fold", "no fold declared", self.name)
        elif not self.fold_spec.group_keys:
            report = report.add("error", "no-fold-keys", "fold has no group keys", self.name)
        return report

    # ---- shared computation interface ------------------------------------

    def has_view(self, view: str) -> bool:
        return self.fold_spec is not None and view == "output"

    def view_schema(self, view: str) -> object | None:
        if self.fold_spec is not None and view == "output":
            return self.fold_spec.output_schema
        return None

    # ---- introspection ---------------------------------------------------

    def _component_name(self) -> str:
        comp = self.component_computation
        return getattr(comp, "name", type(comp).__name__ if comp is not None else "(none)")

    def info(self) -> dict[str, object]:
        return {
            "name": self.name,
            "input": getattr(self.input_schema, "__name__", self.input_schema),
            "component": self._component_name(),
            "component_view": self.component_view,
            "exposure": list(self.exposure_exprs),
            "fold": self.fold_spec.info() if self.fold_spec is not None else None,
        }

    def explain(self) -> str:
        iname = getattr(self.input_schema, "__name__", self.input_schema)
        lines = [f"Structure {self.name}", "", "Input:", f"  - {iname}"]
        lines += ["", "Component computation:"]
        comp = self.component_computation
        kind = "router" if hasattr(comp, "branches") else "graph"
        lines.append(f"  - {kind}: {self._component_name()}")
        lines.append(f"  - view: {self.component_view}")
        schema = comp.view_schema(self.component_view) if comp is not None else None
        if schema is not None:
            lines.append(f"  - output schema: {getattr(schema, '__name__', schema)}")
        branches = getattr(comp, "branches", None)
        if branches is not None:
            lines += ["", "Component branches:"]
            fallback = getattr(comp, "fallback", None)
            if fallback is not None:
                fname = getattr(fallback, "name", "?")
                lines.append(f"  - default -> {fname}.{self.component_view}")
            for branch in branches:
                bname = getattr(branch.computation, "name", "?")
                lines.append(f"  - {branch.label} -> {bname}.{self.component_view}")
        lines += ["", "Exposure:"]
        if self.exposure_exprs:
            lines += [f"  - {name} = {expr}" for name, expr in self.exposure_exprs.items()]
        else:
            lines.append("  - (none)")
        lines += ["", "Fold:"]
        if self.fold_spec is not None:
            lines.append(f"  - group by: {', '.join(self.fold_spec.group_keys)}")
            for name, agg in self.fold_spec.aggregations.items():
                lines.append(f"  - {name} = {agg.describe()}")
            sname = getattr(self.fold_spec.output_schema, "__name__", None)
            if sname:
                lines += ["", "Returns:", f"  - {sname}"]
        else:
            lines.append("  - (none)")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        iname = getattr(self.input_schema, "__name__", "input")
        comp = self._component_name()
        sname = (
            getattr(self.fold_spec.output_schema, "__name__", "output")
            if self.fold_spec is not None
            else "output"
        )
        keys = ", ".join(self.fold_spec.group_keys) if self.fold_spec is not None else ""
        exposure = ", ".join(self.exposure_exprs) or "(identity)"
        return "\n".join(
            [
                "flowchart LR",
                f'    input["{iname}"] --> components["{comp} ({self.component_view})"]',
                f'    components --> exposure["exposure: {exposure}"]',
                f'    exposure --> fold["fold by {keys}"]',
                f'    fold --> output["{sname}"]',
            ]
        )
