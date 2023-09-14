import abc
from typing import Dict, Tuple

from invest_ai.simulation.enums import Decision, Status


class AbstractInvestor(abc.ABC):
    @abc.abstractmethod
    def make_decision(
        self,
        status: Status,
        available_stocks_and_prices: Dict[str, float],
    ) -> Decision:
        """
        Make a decision on how to allocate the available cash to buy stocks.
        :param status: The current status of the bank.
        :param available_stocks_and_prices: A dictionary of available stocks and their prices.
        :return: A decision on how to allocate the available cash to buy stocks, and/or which stocks to sell.
        """
        raise NotImplementedError
