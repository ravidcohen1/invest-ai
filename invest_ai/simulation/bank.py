import datetime
from typing import Dict, List, Optional

from invest_ai.data_collection.finance_store import FinanceStore
from invest_ai.simulation.enums import Status


class Bank:
    def __init__(
        self,
        start_date: datetime.date,
        fs: FinanceStore,
        initial_amount: float = 0.0,
        buying_at: str = "open",
        selling_at: str = "close",
        trading_on_weekend: bool = False,
    ):
        """
        Initialize a Bank object.

        :param start_date: The start date for the bank.
        :param fs: An instance of FinanceStore for financial data.
        :param initial_amount: The initial amount of cash to deposit.
        :param buying_at: Default metric at which to buy ('open', 'close', etc.).
        :param selling_at: Default metric at which to sell ('open', 'close', etc.).
        :param trading_on_weekend: Whether to allow trading on weekends.
        """
        self._date = start_date
        if self.is_weekend():
            raise ValueError(
                f"Start date should not be a weekend. {start_date} is a weekend."
            )
        self.fs = fs
        self._cash = float(initial_amount)
        self.total_deposits = float(initial_amount)
        self._portfolio: Dict[str, int] = {}
        self._history: List[Status] = []
        self.buying_at = buying_at
        self.selling_at = selling_at
        self.trading_on_weekend = trading_on_weekend

    def buy(self, symbol: str, shares: int):
        """
        Buy shares of a symbol.

        :param symbol: The stock symbol to buy.
        :param shares: The number of shares to buy.
        :raises ValueError: If cash is insufficient or number of shares is negative.
        """
        if shares <= 0:
            raise ValueError("Number of shares to buy should be greater than zero.")

        price = self.fs.get_price(symbol, self._date, metric=self.buying_at)
        cost = price * shares
        if self._cash < cost:
            raise ValueError("Insufficient cash to make the purchase.")

        self._cash -= cost
        self._portfolio[symbol] = self._portfolio.get(symbol, 0) + shares

    def sell(self, symbol: str, shares: int):
        """
        Sell shares of a symbol.

        :param symbol: The stock symbol to sell.
        :param shares: The number of shares to sell.
        :raises ValueError: If stock is not owned, insufficient shares are owned, or number of shares is negative.
        """
        if shares <= 0:
            raise ValueError("Number of shares should be greater than zero.")

        if symbol not in self._portfolio:
            raise ValueError(f"You don't own any shares of {symbol}.")

        owned_shares = self._portfolio[symbol]
        if owned_shares < shares:
            raise ValueError("You don't own enough shares to complete the sale.")

        price = self.fs.get_price(symbol, self._date, metric=self.selling_at)
        self._cash += price * shares
        self._portfolio[symbol] -= shares

        if self._portfolio[symbol] == 0:
            del self._portfolio[symbol]

    def daily_update(self) -> Status:
        """
        Update the bank's state at the end of a trading day.
        """
        if len(self._history) > 0:
            assert self._date == self._history[-1].date + datetime.timedelta(
                days=1
            ), "Bank needs to be updated by the end of each day."
        self._history.append(self.get_status())
        self._date += datetime.timedelta(days=1)
        return self.get_status()

    def get_status(self):
        return Status(
            self._date,
            self._cash,
            self._portfolio.copy(),
            total_value=self.get_total_value(),
            total_deposits=self.total_deposits,
        )

    def get_total_value(self, metric=None) -> float:
        """
        Retrieve the total value of the bank.

        :return: The total value of the bank.
        """
        metric = metric or self.selling_at
        total = self._cash
        for symbol, shares in self._portfolio.items():
            total += self.fs.get_price(symbol, self._date, metric=metric) * shares
        return total

    def is_weekend(self) -> bool:
        """
        Check if the current date is a weekend.

        :return: True if the current date is a weekend, False otherwise.
        """
        return self._date.weekday() >= 5

    def deposit(self, amount: float) -> float:
        """
        Deposit an amount into the bank's cash reserves.

        :param amount: The amount to deposit.
        :return: The new cash balance.
        :raises ValueError: If the deposit amount is negative.
        """
        if amount <= 0:
            raise ValueError("Amount to deposit should be greater than zero.")
        self.total_deposits += amount
        self._cash += amount
        return self._cash

    def get_portfolio(self) -> Dict:
        """
        Retrieve a copy of the current stock portfolio.

        :return: A dictionary representing the current stock portfolio.
        """
        return self._portfolio.copy()

    def get_cash(self) -> float:
        """
        Retrieve the current cash balance.

        :return: The current cash balance.
        """
        return self._cash

    def get_history(self) -> List[Status]:
        """
        Retrieve the history of the bank's state.

        :return: A list of dictionaries representing the bank's state history.
        """
        return self._history.copy()

    def get_available_stocks_and_prices(
        self, for_date=None, at=None
    ) -> Optional[Dict[str, float]]:
        """
        Retrieve a list of available stocks and their current open prices.

        :return: A dictionary with stock tickers as keys and their current open prices as values.
        """
        for_date = for_date or self._date
        at = at or self.buying_at
        daily_data = self.fs.get_finance_for_dates(
            for_date, for_date, self.fs.supported_tickers, melt=True
        )
        if len(daily_data) == 0:
            if self.trading_on_weekend:
                return self.get_available_stocks_and_prices(
                    for_date - datetime.timedelta(days=1)
                )
            else:
                return None
        return daily_data.set_index("ticker")[at].to_dict()
