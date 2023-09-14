from datetime import date

import pytest

from invest_ai.investor.always_buy import AlwaysBuyInvestor
from invest_ai.investor.random_investor import RandomInvestor
from invest_ai.simulation.enums import Decision, Status


def test_always_buy_investor():
    always_buy_investor = AlwaysBuyInvestor()
    status = Status(
        date=date(2023, 9, 13),
        cash=1003,
        portfolio={"AAPL": 5},
        total_value=2000,
        total_deposits=1000,
    )
    available_stocks_and_prices = {"AAPL": 150.0}

    decision = always_buy_investor.make_decision(status, available_stocks_and_prices)

    # Assert that the decision is a Decision object
    assert isinstance(decision, Decision), "Should return a Decision object"

    # Assert that the decision is to buy the available stock
    assert "AAPL" in decision.buy, "Should decide to buy AAPL"
    assert decision.buy["AAPL"] == 6, "Should decide to buy 6 AAPL"

    # Assert that the decision does not include any sell orders
    assert decision.sell == {}, "Should not decide to sell any stock"

    # Assert that an AssertionError is raised if more than one stock is available
    multiple_stocks_and_prices = {"AAPL": 150.0, "GOOGL": 2000.0}

    with pytest.raises(AssertionError):
        always_buy_investor.make_decision(status, multiple_stocks_and_prices)


def test_random_investor():
    random_investor = RandomInvestor()
    status = Status(
        date=date(2023, 9, 13),
        cash=1000,
        portfolio={"AAPL": 5},
        total_value=2000,
        total_deposits=1000,
    )
    available_stocks_and_prices = {"AAPL": 150.0, "GOOGL": 1000.0}

    decision_outcomes = set()

    # Run the test multiple times to cover all possible outcomes
    for _ in range(100):
        decision = random_investor.make_decision(status, available_stocks_and_prices)

        # Assert that the decision is a Decision object
        assert isinstance(decision, Decision), "Should return a Decision object"

        # Record whether stocks were bought or not
        decision_outcomes.add(bool(decision.buy))

    # Assert that both outcomes are possible (buying and not buying)
    assert True in decision_outcomes, "Should sometimes decide to buy stocks"
    assert False in decision_outcomes, "Should sometimes decide not to buy stocks"

    random_investor = RandomInvestor()
    status = Status(
        date=date(2023, 9, 13), cash=1, portfolio={}, total_value=1, total_deposits=0
    )
    available_stocks_and_prices = {"AAPL": 150.0, "GOOGL": 1000.0}
    for _ in range(100):
        decision = random_investor.make_decision(status, available_stocks_and_prices)
        assert decision.buy == {}, "Should not buy any stock if cash is insufficient"
