"""Run the inflation-linked energy forward end to end.

Run with:  uv run python -m examples.custom_instrument.run

Two contracts that differ in index AND in the calendar rule used to read the
index factor:

- ENG-IPCA: tenor Jun/2029, IPCA  -> factor read at 2029-01-01
- ENG-CPI:  tenor Jun/2028, CPI   -> factor read at 2028-04-01

One graph prices both; the convention registry + the reference_date join key do
the selection.
"""

from __future__ import annotations

from datetime import date

import polars as pl
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource

from .conventions import add_reference_date
from .pricer import price_inflation_energy

market = MarketSnapshot.from_sources(
    as_of=date(2026, 6, 3),
    sources=[
        # Annual projected-index points, each at its index's convention date.
        MarketSource(
            "inflation_curve",
            pl.DataFrame(
                {
                    "id_indexador": [20, 20, 10, 10],
                    "ref_date": [
                        date(2028, 1, 1),
                        date(2029, 1, 1),  # IPCA Jan-1 points
                        date(2028, 4, 1),
                        date(2029, 4, 1),  # CPI April points
                    ],
                    "projected_index": [112.0, 118.0, 105.0, 108.0],
                }
            ).lazy(),
        ),
        MarketSource(
            "inflation_fixings",
            pl.DataFrame({"id_indexador": [20, 10], "base_index": [100.0, 100.0]}).lazy(),
        ),
        MarketSource(
            "di_curve",
            pl.DataFrame({"tenor_days": [504, 756], "zero_rate": [0.10, 0.10]}).lazy(),
        ),
        MarketSource("fx_rates", pl.DataFrame({"currency": ["BRL"], "fx_rate": [1.0]}).lazy()),
    ],
)

legs = pl.DataFrame(
    {
        "instrument_id": ["ENG-IPCA", "ENG-CPI"],
        "id_indexador": [20, 10],
        "tenor_date": [date(2029, 6, 1), date(2028, 6, 1)],
        "payment_days": [756, 504],
        "forward_price": [130.0, 130.0],
        "strike": [100.0, 100.0],
        "currency": ["BRL", "BRL"],
    }
).lazy()

# Show the convention-derived reference_date, then the priced output.
print("Reference dates picked per index convention:")
print(
    add_reference_date(legs)
    .select("instrument_id", "id_indexador", "tenor_date", "reference_date")
    .collect()
)
print("\nPriced instruments:")
print(price_inflation_energy(legs, market).collect())
