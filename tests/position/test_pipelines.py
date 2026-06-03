from __future__ import annotations

import pytest
from schenberg.domain.schemas.position import Position
from schenberg.position.pipelines import valuation_pipe


def test_valuation_pipe_exposes_simplified_position_pricing_stages(energy_inputs, energy_market):
    positions = Position.from_records(
        [
            {
                "position_id": "POS-1",
                "book": "Energy Desk",
                "instrument_type": "FORWARD",
                "instrument_id": "ENG-1",
                "quantity": 100.0,
                "side": 1.0,
            }
        ]
    )

    env = valuation_pipe.run(forwards=energy_inputs, positions=positions, market=energy_market)

    assert {"forward_prices", "priced_positions", "book_mtm"}.issubset(env)
    assert valuation_pipe.order() == ["forward_prices", "priced_positions", "book_mtm"]
    assert env["book_mtm"].collect().select("mtm").item() == pytest.approx(4883.174188, rel=1e-6)
