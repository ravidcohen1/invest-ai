import datetime
from typing import List

from invest_ai.data_collection.finance_store import FinanceStore
from invest_ai.investor.abstract_investor import AbstractInvestor
from invest_ai.simulation.bank import Bank, Status


class Simulator:
    """
    A simulator to perform stock trading simulations.

    This class simulates daily stock trading using an investor model and a bank model.
    """

    def __init__(
        self, bank: Bank, investor: AbstractInvestor, monthly_budget: float = 0.0
    ):
        """
        Initialize the Simulator object.
        :param bank: The Bank object to handle transactions.
        :param investor: The AbstractInvestor object to make decisions.
        :param monthly_budget: The monthly budget to deposit into the bank. Default is 0.0.
        """
        self.bank = bank
        self.investor = investor
        self.start_date = bank.get_status().date
        self.monthly_budget = monthly_budget

    def next_day(self) -> Status:
        """
        Simulate the activities for the next trading day.

        :return: The updated status after performing the day's activities.
        :raises ValueError: If new_budget is negative.
        """

        status = self.bank.get_status()
        value_before_trading = self.bank.get_total_value()

        if status.date.day == 1:
            if self.monthly_budget > 0:
                self.bank.deposit(self.monthly_budget)
            status = self.bank.get_status()
        available_stocks = self.bank.get_available_stocks_and_prices()
        if available_stocks is None:
            new_status = self.bank.daily_update()
            value_after_trading = self.bank.get_total_value()
            if value_after_trading < value_before_trading:
                print()
                print("day", status.date.weekday())
                print("SPY", new_status.portfolio["SPY"])
                print()
            return new_status
        decision = self.investor.make_decision(status, available_stocks)

        for stock, amount in decision.sell.items():
            self.bank.sell(stock, amount)

        for stock, amount in decision.buy.items():
            self.bank.buy(stock, amount)

        new_status = self.bank.daily_update()

        value_after_trading = self.bank.get_total_value()
        if value_after_trading < value_before_trading:
            print()
            print("sell", decision.sell)
            print("buy", decision.buy)
            print("day", status.date.weekday())
            print("SPY", new_status.portfolio["SPY"])
            decision = self.investor.make_decision(status, available_stocks)
            v = self.bank.get_total_value()
        return new_status

    def finalize(self) -> List[Status]:
        """
        Finalize the simulation and return the trading history.

        :return: A list of status objects representing the trading history.
        """
        return self.bank.get_history()


if __name__ == "__main__":
    fs = FinanceStore()
    bank = Bank(initial_amount=1000, start_date=datetime.date(2023, 1, 1), fs=fs)
