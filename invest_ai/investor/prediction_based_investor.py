from typing import Dict, List

import pandas as pd

from invest_ai.investor.abstract_investor import AbstractInvestor
from invest_ai.return_predictor.baselines import AbstractReturnPredictor
from invest_ai.simulation.enums import Decision, Status


class PredictionBasedInvestor(AbstractInvestor):
    """
    An investor that makes decisions based on a given return predictor.

    This class assumes that only one stock is available for buying or selling.
    """

    def __init__(
        self,
        predictor: AbstractReturnPredictor,
        buy_labels: List[str],
        sell_labels: List[str],
    ):
        """
        Initialize the PredictionBasedInvestor object.

        :param predictor: The return predictor to use for making decisions.
        """
        self.predictor = predictor
        self.buy_labels = buy_labels
        self.sell_labels = sell_labels

    def make_decision(
        self, status: Status, available_stocks_and_prices: Dict[str, float]
    ) -> Decision:
        """
        Make a trading decision based on the current status and available stocks.

        :param status: Current status containing the available cash and portfolio.
        :param available_stocks_and_prices: Dictionary containing available stocks and their prices.
        :return: A Decision object containing the stocks to buy or sell.
        :raises ValueError: If more than one stock is available.
        """
        if len(available_stocks_and_prices) != 1:
            raise ValueError("Only one stock is supported")

        stock = list(available_stocks_and_prices.keys())[0]
        price = list(available_stocks_and_prices.values())[0]

        prediction = self.predictor.predict(
            pd.Series({"stock": stock, "date": status.date})
        )

        if prediction in self.buy_labels:
            amount = int(status.cash / price)
            if amount > 0:
                return Decision(buy={stock: amount})
        elif prediction in self.sell_labels:
            if stock in status.portfolio:
                return Decision(sell={stock: status.portfolio[stock]})

        return Decision()
