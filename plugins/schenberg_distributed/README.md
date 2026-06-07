# schenberg-distributed

`schenberg-distributed` adds execution boundaries and whole-node valuation DAGs
without changing the Schenberg core. The core remains pure and lazy: FormulaGraph
continues to compile to Polars expressions, and this plugin handles only plan
orchestration and materialization.

## ValuationPlan

`ValuationPlan` is a small `rustworkx`-backed DAG. Edges point from dependency to
dependent, and the distribution grain is a complete valuation node (a callable
that receives inputs and returns a `polars.LazyFrame` or `polars.DataFrame`). The
plugin never distributes internal formula terms such as `T`, `DF`, or `PV`.

```python
from schenberg_distributed import LocalExecutor, ValuationPlan

plan = ValuationPlan("book")
plan.input("trades", trades)
plan.input("market", market)

@plan.node("forward_values", trades="trades", market="market", market_sources=("curves",))
def forward_values(trades, market):
    return price_forward(trades, market)

executor = LocalExecutor()
lazy_frame = executor.lazy(plan, target="forward_values")
data = executor.collect(plan, target="forward_values")
```

The decorator records metadata and returns the original function. Building a plan
never executes the decorated function.

## Executors

- `LocalExecutor.lazy(plan, target=...)` runs the DAG in-process and preserves
  `LazyFrame` outputs whenever possible.
- `LocalExecutor.collect(plan, target=...)` materializes the final target through
  `PricingExecutionContext`.
- `DaskExecutor.collect(plan, target=...)` is an optional MVP executor using
  `dask.delayed`, one task per valuation node. Install with
  `schenberg-distributed[dask]`.

## Existing pricing contexts

The original API is still available:

```python
from schenberg_distributed import PricingExecutionContext, collect_pricing

context = PricingExecutionContext.local(engine="streaming")
result = collect_pricing(lazy_pricing_frame, context=context)
```

Ray support is optional via `schenberg-distributed[ray]`.
