import datetime

import pytest

from invest_ai.data_collection.finance_store import FinanceStore
from invest_ai.investor.random_investor import RandomInvestor
from invest_ai.simulation.bank import Bank, Status
from invest_ai.simulation.simulator import Simulator


@pytest.fixture(scope="function")
def simulator(finance_store: FinanceStore) -> Simulator:
    bank = Bank(
        start_date=datetime.date(2023, 8, 15), fs=finance_store, initial_amount=10000
    )
    investor = RandomInvestor()
    return Simulator(bank=bank, investor=investor, monthly_budget=1000)


def test_constructor(simulator: Simulator):
    assert simulator is not None, "Simulator should be properly initialized"


def test_next_day(simulator: Simulator):
    initial_status = simulator.next_day()  # Assuming that next_day returns the status

    # Assert that the status is a Status object
    assert isinstance(initial_status, Status), "Should return a Status object"

    new_status = simulator.next_day()
    # Assert that the new status is updated
    assert new_status.date == initial_status.date + datetime.timedelta(
        days=1
    ), "Date should be updated"


def test_finalize(simulator: Simulator):
    simulator.next_day()
    simulator.next_day()
    simulator.next_day()

    # todo test monthly budget
    history = simulator.finalize()

    # Assert that the history is a list of Status objects
    assert isinstance(history, list), "Should return a list"
    assert len(history) > 0, "List should not be empty"
    assert all(
        isinstance(status, Status) for status in history
    ), "All elements in the list should be Status objects"
