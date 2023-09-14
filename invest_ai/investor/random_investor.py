import random
from typing import Dict

from invest_ai.investor.abstract_investor import AbstractInvestor
from invest_ai.return_predictor.baselines import RandomReturnPredictor
from invest_ai.simulation.enums import Decision, Status


class RandomInvestor(AbstractInvestor):
    def __init__(self):
        self.predictor = RandomReturnPredictor(labels=["bad", "neutral", "good"])

    def make_decision(
        self, status: Status, available_stocks_and_prices: Dict[str, float]
    ) -> Decision:
        stocks_to_buy = []
        for s in available_stocks_and_prices:
            prediction = self.predictor.predict(None)
            if prediction == "good":
                stocks_to_buy.append(s)
        if len(stocks_to_buy) == 0:
            return Decision()
        chosen_stock = random.choice(stocks_to_buy)
        amount = int(status.cash / available_stocks_and_prices[chosen_stock])
        if amount == 0:
            return Decision()
        else:
            return Decision(buy={chosen_stock: amount})
