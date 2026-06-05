from __future__ import annotations

from schenberg.domain.rules import rule_for
from schenberg.market_data import date_rules as dates
from schenberg.pricing.instruments.derivatives.forwards.contracts import (
    ForwardContractPricing,
)


class EnergyForwardPricing(ForwardContractPricing):
    """Energy forward contract schema.

    Energy forwards reuse the generic forward formula. Extra columns here are
    contract/market-routing coordinates, not new pricing math.

    The PLD index overrides the default fixing rule: the fixing date is the
    6th business day of the month following the delivery period (tenor month).
    """

    submarket: str
    incentive: str
    delivery_period: str

    @rule_for("index_fixing_date", selector="indexer", value="PLD")
    def _pld(cls):  # noqa: N805
        return dates.nth_business_day_next_month("tenor", n=6)
