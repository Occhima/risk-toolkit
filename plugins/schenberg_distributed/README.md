# schenberg-distributed

`schenberg-distributed` is a project-extension package in the Schenberg uv
workspace. It adds execution contexts for pricing graph computations without
changing the lazy pricing APIs in `schenberg`.

The plugin keeps graph construction lazy, then centralizes materialization in a
small `PricingExecutionContext` object. The default local backend forwards
keyword arguments to `polars.LazyFrame.collect`. The Ray backend optionally
initializes Ray before collection, making it a safe integration point for Ray's
Polars execution support as it evolves. Additional backend hooks can wrap custom
engines while preserving the same pricing call surface.

## Example

```python
from schenberg_distributed import PricingExecutionContext, collect_pricing

context = PricingExecutionContext.ray(streaming=True, engine="streaming")
result = collect_pricing(lazy_pricing_frame, context=context)
```
