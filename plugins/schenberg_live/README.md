# schenberg-live

`schenberg-live` is a synchronous, in-memory event layer for
`schenberg_distributed.ValuationPlan`. It is deliberately small: no Kafka, Redis,
WebSocket, server, scheduler, async loop, daemon, or background thread.

- `MarketEvent` describes a market source update and version.
- `PositionEvent` describes a position/book update and version.
- `DependencyIndex` asks the valuation plan which nodes are affected.
- `ValuationCache` stores results by `(target, version)`.
- `LiveValuationEngine` invalidates affected targets, executes a plan with an
  executor, and returns `LiveResult`.

Repeated calls with the same version return `cache_hit=True`; new versions
recompute synchronously in the caller thread.

```python
from schenberg_distributed import LocalExecutor, ValuationPlan
from schenberg_live import LiveValuationEngine, MarketEvent

engine = LiveValuationEngine(plan=plan, executor=LocalExecutor(), target="forward_values")
result = engine.on_market_event(MarketEvent(source="curves", version="mkt-001"))
```
