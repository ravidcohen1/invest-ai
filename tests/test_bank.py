import datetime

import numpy as np
import pytest

from invest_ai.simulation.bank import Bank, Status

# ... (other imports and setup)


# No changes are required for test_init
def test_init(bank: Bank) -> None:
    assert bank is not None, "Bank should be initialized properly"


# Update test_buy
def test_buy(bank: Bank) -> None:
    with pytest.raises(ValueError):
        bank.buy("AAPL", 1000)

    with pytest.raises(ValueError):
        bank.buy("AAPL", -1)

    initial_cash = bank.get_cash()
    expected_cost = (
        bank.get_available_stocks_and_prices()["AAPL"] * 10
    )  # Use instance variable
    initial_portfolio = bank.get_portfolio()

    bank.buy("AAPL", 10)

    # Assert cash and portfolio changes
    assert np.isclose(
        bank.get_cash(), initial_cash - expected_cost
    ), "Cash was not deducted properly"
    assert "AAPL" in bank.get_portfolio(), "Purchase is not reflected in portfolio"
    assert (
        bank.get_portfolio()["AAPL"] == initial_portfolio.get("AAPL", 0) + 10
    ), "Purchase is not reflected in portfolio"


# Update test_sell
def test_sell(bank: Bank) -> None:
    with pytest.raises(ValueError):
        bank.sell("GOOGL", 1)

    with pytest.raises(ValueError):
        bank.sell("AAPL", -1)

    bank.buy("AAPL", 10)

    initial_cash = bank.get_cash()
    initial_portfolio = bank.get_portfolio()
    expected_gain = (
        bank.get_available_stocks_and_prices(at=bank.selling_at)["AAPL"] * 5
    )  # Use instance variable

    bank.sell("AAPL", 5)

    # Assert cash and portfolio changes
    assert np.isclose(
        bank.get_cash(), initial_cash + expected_gain
    ), "Cash was not added properly"
    assert (
        bank.get_portfolio()["AAPL"] == initial_portfolio["AAPL"] - 5
    ), "Portfolio was not updated properly"


# ... (rest of the tests remain unchanged)


def test_daily_update(bank: Bank):
    initial_date = bank._date  # Assuming you have a 'date' attribute in your Bank class
    initial_history_length = len(
        bank._history
    )  # Assuming _history is a list attribute in your Bank class

    new_date = initial_date + datetime.timedelta(days=1)

    status = bank.daily_update()

    # Assert that the internal date attribute is updated
    assert bank._date == new_date, "Internal date should be updated to the new date"

    # Assert that the history length has increased by 1
    assert (
        len(bank._history) == initial_history_length + 1
    ), "History should have one more entry"

    # Assert that the returned status is the last item in history
    assert (
        status == bank._history[-1]
    ), "Returned status should be the last item in history"


def test_deposit(bank: Bank) -> None:
    initial_cash = bank.get_cash()

    with pytest.raises(ValueError):
        bank.deposit(-1)

    new_cash = bank.deposit(100)
    assert new_cash == initial_cash + 100, "Cash should increase by 100 after deposit"


def test_get_portfolio(bank: Bank) -> None:
    assert isinstance(
        bank.get_portfolio(), dict
    ), "Should return a dictionary representing the current stock portfolio"


def test_get_cash(bank: Bank) -> None:
    assert isinstance(bank.get_cash(), float), "Should return the current cash balance"


def test_get_history(bank: Bank) -> None:
    initial_history = bank.get_history()
    bank.daily_update()
    history = bank.get_history()
    assert isinstance(
        history, list
    ), "Should return a list of dictionaries representing the bank's state history"
    assert (
        len(history) == len(initial_history) + 1
    ), "Should append new status to history"
    for s in history:
        assert isinstance(
            s, Status
        ), "Should return a list of dictionaries representing the bank's state history"
        assert isinstance(
            s.date, datetime.date
        ), "Should return a list of dictionaries representing the bank's state history"
        assert isinstance(
            s.cash, float
        ), "Should return a list of dictionaries representing the bank's state history"
        assert isinstance(
            s.portfolio, dict
        ), "Should return a list of dictionaries representing the bank's state history"


def test_get_available_stocks_and_prices(bank: Bank) -> None:
    assert isinstance(
        bank.get_available_stocks_and_prices(), dict
    ), "Should return a dictionary with stock tickers and their current open prices"
