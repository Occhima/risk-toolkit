# Concepts

Schenberg is a small lazy pricing DSL over Polars expressions.

## Terms and formula graphs

A `FormulaGraph` is a directed acyclic graph of `Term`s:

- input terms are columns supplied by the trade frame;
- market terms are columns attached from a `MarketSnapshot`;
- formula terms are `pl.Expr` functions over earlier terms;
- views map graph terms into output schemas.

`compute(frame, market=..., view=...)` returns a `pl.LazyFrame`. It validates that
required input and market join-key columns exist, attaches market data, and adds
view expressions. It does not collect trade rows.

`stage(..., allow_missing=True)` is only for debugging. Normal `compute()` and
`stage()` fail when required inputs are missing.

## Market data

A `MarketSnapshot` is the environment for market reads. `MarketSource` can carry
`unique_by=(...)` to validate quote-key uniqueness once during snapshot
construction. This validation may collect market source data; that is an explicit
market-data boundary, not trade pricing.

Keyed requirements perform lazy joins. Interpolated requirements may build an
in-memory interpolation book from quote grids first, then return a lazy trade-side
frame whose interpolation expression runs when the caller collects.

## Pricing boundary

Pure pricing functions return own-currency values. They do not read book columns,
position side, quantity, legal entity, reporting currency, or reported MTM.

Implemented public pricing functions are:

- `price_forward`
- `forward_instrument_value`
- `price_energy_forward`
- `energy_forward_instrument_value`

Generic and energy forwards share the same forward formula builder; energy
forwards override only the contract schema and market requirements.

## Position boundary

The position layer consumes already-valued instruments:

- `InstrumentValue.value` is in `InstrumentValue.currency`;
- `Position.side * Position.quantity` creates exposure;
- `BookContract.reporting_currency` and `ReportingFx.book_fx` convert reported
  MTM.

Aggregation is explicit and uses `Fold` after position values are computed.
