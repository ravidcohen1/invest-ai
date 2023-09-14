from typing import Dict

from invest_ai.investor.abstract_investor import AbstractInvestor
from invest_ai.simulation.enums import Decision, Status


class NeverBuyInvestor(AbstractInvestor):
    def make_decision(
        self, status: Status, available_stocks_and_prices: Dict[str, float]
    ) -> Decision:
        return Decision()
