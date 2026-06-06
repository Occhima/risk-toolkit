# Energy forward and formula reuse

Energy forwards reuse the generic forward formula builder and override only the
contract schema plus market requirements:

```python
from schenberg.pricing.instruments.derivatives.forwards.energy.api import energy_forward_formula
from schenberg.pricing.instruments.derivatives.forwards.energy.market import EnergyForwardMarket

assert energy_forward_formula.name == "energy_forward"
assert EnergyForwardMarket.__requirements__["forward_price"].table == "energy_forward_curve"
```

Use `price_energy_forward(trades, market)` for pure own-currency pricing or
`energy_forward_instrument_value(trades, market)` to feed the position layer.
