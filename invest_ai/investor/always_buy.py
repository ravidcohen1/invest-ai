from typing import Dict

from invest_ai.investor.abstract_investor import AbstractInvestor
from invest_ai.simulation.enums import Decision, Status


class AlwaysBuyInvestor(AbstractInvestor):
    """
    An investor that always buys the available stock, as much as the budget allows.

    This class assumes that only one stock is available for buying.
    """

    def make_decision(
        self, status: Status, available_stocks_and_prices: Dict[str, float]
    ) -> Decision:
        """
        Make a decision to buy the available stock.

        :param status: Current status containing the available cash.
        :param available_stocks_and_prices: Dictionary containing available stocks and their prices.
        :return: A Decision object containing the stock to buy and the amount.
        :raises AssertionError: If more than one stock is available.
        """
        budget = status.cash
        assert (
            len(available_stocks_and_prices) == 1
        ), "Only one stock should be available."
        stock, price = list(available_stocks_and_prices.items())[0]
        amount = int(budget / price)
        if amount == 0:
            return Decision()
        else:
            return Decision(buy={stock: amount})
