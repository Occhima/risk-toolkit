"""The position layer: value, PnL explain, and risk — one view shape, three quantities.

Pricing returns a **pure** per-instrument quantity (no ``side``, no position): a
single ``InstrumentValue``, a PnL decomposition (``InstrumentPnlExplain``), or a
vector of risk factors (``InstrumentRisk`` — the Greeks). A ``PositionView`` lifts
*any* of them onto a ``Position`` and a book/reporting context, exposing the
measures by name. ``side`` enters only here; reporting-currency conversion is a
measure (``/ book_fx``), never a pricing concern.

Run me::

    uv run python examples/position_layer.py
"""

from __future__ import annotations

import polars as pl
from schenberg.domain.schemas.position import (
    BookContract,
    InstrumentPnlExplain,
    InstrumentRisk,
    InstrumentValue,
    Position,
    ReportingFx,
)
from schenberg.position import position_pnl_explain, position_risk, position_value

# One book of two positions: a long USD forward and a short USD option.
positions = Position.from_records(
    [
        {
            "position_id": "P-FWD",
            "book": "RATES",
            "instrument_type": "FORWARD",
            "instrument_id": "FWD-1",
            "quantity": 100.0,
            "side": 1.0,  # long
            "unit_notional": 1_000.0,
        },
        {
            "position_id": "P-OPT",
            "book": "RATES",
            "instrument_type": "OPTION",
            "instrument_id": "OPT-1",
            "quantity": 10.0,
            "side": -1.0,  # short
            "unit_notional": None,
        },
    ]
)

# Book context: desk / entity / reporting currency live here, joined once by `book`.
book = BookContract.from_records(
    [{"book": "RATES", "desk": "Macro", "legal_entity": "LE-BR", "reporting_currency": "BRL"}]
)
# Convert each instrument currency into the book's reporting currency (BRL).
fx = ReportingFx.from_records(
    [{"currency": "USD", "reporting_currency": "BRL", "book_fx": 0.2}]  # 1 USD = 5 BRL
)


def main() -> None:
    # ---- 1. position value: exposure, notional, mtm, reported mtm ----
    values = InstrumentValue.from_records(
        [
            {
                "instrument_type": "FORWARD",
                "instrument_id": "FWD-1",
                "value": 9.05,
                "currency": "USD",
            },
            {
                "instrument_type": "OPTION",
                "instrument_id": "OPT-1",
                "value": 4.20,
                "currency": "USD",
            },
        ]
    )
    print("position_value\n", position_value(positions, value=values, book=book, fx=fx).collect())

    # ---- 2. position PnL explain: each component lifted to the book, additive ----
    pnl = InstrumentPnlExplain.from_records(
        [
            {
                "instrument_type": "FORWARD",
                "instrument_id": "FWD-1",
                "currency": "USD",
                "roll_value_pnl": 0.10,
                "curve_value_pnl": 0.30,
                "fx_value_pnl": 0.05,
                "fixing_value_pnl": 0.00,
                "residual_value_pnl": 0.01,
                "total_value_pnl": 0.46,
            },
            {
                "instrument_type": "OPTION",
                "instrument_id": "OPT-1",
                "currency": "USD",
                "roll_value_pnl": -0.02,
                "curve_value_pnl": 0.04,
                "fx_value_pnl": 0.01,
                "fixing_value_pnl": 0.00,
                "residual_value_pnl": 0.00,
                "total_value_pnl": 0.03,
            },
        ]
    )
    print(
        "\nposition_pnl_explain\n",
        position_pnl_explain(positions, pnl=pnl, book=book, fx=fx).collect(),
    )

    # ---- 3. position risk: each pure Greek scaled by exposure (side flips signs) ----
    risk = InstrumentRisk.from_records(
        [
            {
                "instrument_type": "FORWARD",
                "instrument_id": "FWD-1",
                "currency": "USD",
                "delta": 1.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": -0.01,
                "rho": 0.85,
            },
            {
                "instrument_type": "OPTION",
                "instrument_id": "OPT-1",
                "currency": "USD",
                "delta": 0.6,
                "gamma": 0.02,
                "vega": 12.0,
                "theta": -3.0,
                "rho": 8.0,
            },
        ]
    )
    print("\nposition_risk\n", position_risk(positions, risk=risk).collect())

    # ---- 4. one declaration, many interpretations ----
    print("\nposition_value.explain()\n", position_value.explain())
    print("\nposition_risk.to_mermaid()\n", position_risk.to_mermaid())

    # ---- 5. the stage view materializes every join column + intermediate measure ----
    staged = position_value.stage(positions, value=values, book=book, fx=fx)
    print("\nstage() columns:", staged.collect_schema().names())


if __name__ == "__main__":
    pl.Config.set_tbl_cols(-1)
    main()
