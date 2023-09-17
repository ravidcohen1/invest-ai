import datetime
from typing import Dict


class Status:
    """
    Represents the current status of a bank.
    """

    def __init__(
        self,
        date: datetime.date,
        cash: float,
        portfolio: Dict[str, int],
        total_value: float,
        total_deposits: float,
    ):
        """
        Initialize the Status object.

        :param date: The current date.
        :param cash: The current cash available.
        :param portfolio: The current stock portfolio.
        """
        self.date = date
        self.cash = cash
        self.portfolio = portfolio
        self.total_value = total_value
        self.total_deposits = total_deposits

    def get_total_profit(self) -> float:
        return self.total_value - self.total_deposits

    def get_profit_percentage(self) -> float:
        return self.get_total_profit() / self.total_deposits

    def to_dict(self):
        return {
            "cash": self.cash,
            "portfolio": self.portfolio,
            "total_value": self.total_value,
            "total_deposits": self.total_deposits,
            "total_profit": self.get_total_profit(),
            "profit_percentage": self.get_profit_percentage(),
        }


class Decision:
    """
    Represents the decision made by an investor.
    """

    def __init__(self, buy: Dict[str, int] = None, sell: Dict[str, int] = None):
        """
        Initialize the Decision object.

        :param buy: Dictionary containing stocks to buy and their quantities.
                    Default is an empty dictionary.
        :param sell: Dictionary containing stocks to sell and their quantities.
                     Default is an empty dictionary.
        """
        self.buy = buy or {}
        self.sell = sell or {}
        self.buy = {k: v for k, v in self.buy.items() if v > 0}
        self.sell = {k: v for k, v in self.sell.items() if v > 0}
