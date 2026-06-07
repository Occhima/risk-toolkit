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
  `dask.delayed`, one task per whole valuation node. It only helps when
  user-defined nodes/partitions are coarse enough; it does not distribute one
  Polars `LazyFrame` query internally. Install with `schenberg-distributed[dask]`.
- `PartitionedPricingPlan` plus `collect_partitioned_local(...)` is an explicit
  helper for eager trade partitions: the same pricer is collected per partition
  and concatenated.

## Existing pricing contexts

The original API is still available:

```python
from schenberg_distributed import PricingExecutionContext, collect_pricing

context = PricingExecutionContext.local(engine="streaming")
result = collect_pricing(lazy_pricing_frame, context=context)
```

Ray support is optional via `schenberg-distributed[ray]`. The built-in Ray
context initializes Ray before handing the `LazyFrame` to local Polars
`collect`; register a custom backend if you need different materialization
behavior.
