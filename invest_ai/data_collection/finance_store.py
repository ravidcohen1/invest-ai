import datetime
import os
from collections.abc import Sequence
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import yfinance as yf

from invest_ai import configs as cfg

YEARLY_10_PERCENT = "YEARLY_10_PERCENT"


class FinanceStore:
    """
    A class to manage the storage and retrieval of finance data and their metadata.
    """

    def __init__(
        self,
        min_date: datetime.date = cfg.MIN_SUPPORTED_DATE,
        max_date: datetime.date = cfg.MAX_SUPPORTED_DATE,
        supported_tickers: Sequence[str] = cfg.SUPPORTED_TICKERS,
        data_path: Path = cfg.FINANCE_DATA_PATH,
    ):
        self._validate_init_input(min_date, max_date, supported_tickers, data_path)
        self.data_path = data_path
        self.min_date = min_date
        self.max_date = max_date
        self.supported_tickers = supported_tickers
        if YEARLY_10_PERCENT in supported_tickers:
            self.df = self._generate_10_percent_stock()
        else:
            self.df = self._load_data()

    def _generate_10_percent_stock(self):
        # Convert start_date and end_date to pandas Timestamp objects
        start_date = self.min_date
        end_date = self.max_date

        # Create a date range from start_date to end_date
        dates = pd.date_range(start=start_date, end=end_date, freq="B")

        # Calculate the daily growth rate for 10% yearly increase
        yearly_rate = 0.1
        daily_rate = (1 + yearly_rate) ** (1 / 252) - 1  # 252 trading days in a year

        # Generate the values for the new ticker "YEARLY_10_PERCENT"
        initial_value = 1.0
        values = initial_value * (1 + daily_rate) ** np.arange(len(dates))

        # Reshape the values array to fit into the DataFrame
        reshaped_values = np.tile(values.reshape(-1, 1), 5)

        # Create a DataFrame for the new ticker with a two-level column index
        cols = pd.MultiIndex.from_product(
            [["Open", "Close", "Min", "Max", "Adj Close"], [YEARLY_10_PERCENT]]
        )
        new_ticker_df = pd.DataFrame(reshaped_values, index=dates, columns=cols)
        new_ticker_df.index.name = "Date"
        return new_ticker_df

    def _load_data(self) -> pd.DataFrame:
        """
        Load finance data from the PKL file into a DataFrame.

        :return: DataFrame containing finance data.
        :rtype: pd.DataFrame
        """
        if os.path.exists(self.data_path):
            data = pd.read_pickle(self.data_path)
        else:
            data = self._download_stock_prices(
                tickers=self.supported_tickers,
                start_date=self.min_date,
                end_date=self.max_date,
            )
            os.makedirs(self.data_path.parent, exist_ok=True)
            data.to_pickle(str(self.data_path))
        available_tickers = data.columns.get_level_values(1).unique()
        assert all(ticker in available_tickers for ticker in self.supported_tickers)
        available_dates = data.index.unique()
        assert self.min_date >= available_dates.min().date()
        assert self.max_date <= available_dates.max().date()

        data = data.loc[
            pd.IndexSlice[self.min_date : self.max_date],
            pd.IndexSlice[:, self.supported_tickers],
        ]
        return data

    @staticmethod
    def _download_stock_prices(
        tickers: Sequence[str], start_date: datetime.date, end_date: datetime.date
    ) -> pd.DataFrame:
        """
        Download stock prices for given tickers and date range.

        :param tickers: List of stock tickers to download.
        :type tickers: List[str]
        :param start_date: Start date of the data download range.
        :type start_date: datetime.date
        :param end_date: End date of the data download range.
        :type end_date: datetime.date
        :return: DataFrame containing downloaded stock price data.
        :rtype: pd.DataFrame
        """
        data = yf.download(
            tickers=tickers, start=start_date, end=end_date, ignore_tz=True
        )
        return data

    def get_finance_for_dates(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        stock_tickers: Union[Sequence[str], str],
        melt=False,
    ) -> pd.DataFrame:
        """
        Fetch or retrieve finance data for specific stock(s) within a date range.

        :param start_date: Start date as datetime.date object.
        :type start_date: datetime.date
        :param end_date: End date as datetime.date object.
        :type end_date: datetime.date
        :param stock_tickers: Stock ticker symbol(s) (e.g., AAPL for Apple Inc.).
        :type stock_tickers: Union[List[str], str]

        :return: DataFrame containing finance data for the specific stock(s) and time periods.
        :rtype: pd.DataFrame
        """
        self._validate_get_finance_for_dates_input(start_date, end_date, stock_tickers)

        # Slice the DataFrame to match the date range
        df = self.df.loc[pd.IndexSlice[start_date:end_date], :]

        # If stock_tickers is a string, convert to a single-item list
        if isinstance(stock_tickers, str):
            stock_tickers = [stock_tickers]

        # Filter by stock_tickers
        df = df.loc[:, pd.IndexSlice[:, stock_tickers]]

        if melt:
            # Reset the index to make 'Date' a column
            df.reset_index(inplace=True)

            # Melt the DataFrame to have 'ticker' and 'metric' as columns
            df_melt = df.melt(
                id_vars=["Date"], var_name=["metric", "ticker"], value_name="value"
            )

            # Create a new DataFrame by pivoting 'df_melt'
            df_pivot = df_melt.pivot(
                index=["Date", "ticker"], columns="metric", values="value"
            )

            # Reset the index for the new DataFrame
            df_pivot.reset_index(inplace=True)

            # Make the metric names lowercase
            df_pivot.columns = [
                col.lower().replace(" ", "_") for col in df_pivot.columns
            ]
            df_pivot = df_pivot.dropna()
            df = df_pivot

        return df

    def get_price(
        self, symbol: str, date: datetime.date, metric: str = "close"
    ) -> float:
        """
        Fetch the price of a stock on a specific date.

        :param symbol: Stock ticker symbol (e.g., AAPL for Apple Inc.).
        :type symbol: str
        :param date: Date to fetch the stock price.
        :type date: datetime.date
        :param metric: Finance metric to fetch (e.g., 'close' for closing price).
        :type metric: str

        :return: Stock price for the given stock and date.
        :rtype: float
        """
        metric = metric.lower()
        if date < self.min_date or date > self.max_date:
            raise ValueError(
                f"date must be between {self.min_date} and {self.max_date}"
            )
        if symbol not in self.supported_tickers:
            raise ValueError(f"symbol {symbol} is not supported")
        if metric not in ["open", "high", "low", "close", "adj_close", "volume"]:
            raise ValueError(f"metric {metric} is not supported")
        df = self.get_finance_for_dates(date, date, symbol, melt=True)
        if len(df) == 0:
            return self.get_price(symbol, date - datetime.timedelta(days=1), metric)
        return df[metric].values[0]

    def _validate_init_input(
        self,
        min_date: datetime.date,
        max_date: datetime.date,
        supported_tickers: Sequence[str],
        data_path: Path,
    ):
        if not isinstance(min_date, datetime.date) or not isinstance(
            max_date, datetime.date
        ):
            raise TypeError("min_date and max_date must be datetime.date objects")
        if min_date > max_date:
            raise ValueError("min_date should be less than or equal to max_date")
        if not supported_tickers or not all(
            isinstance(ticker, str) for ticker in supported_tickers
        ):
            raise ValueError("supported_tickers must be a non-empty list of strings")
        if not isinstance(data_path, Path):
            raise TypeError("data_path must be a pathlib.Path object")

    def _validate_get_finance_for_dates_input(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        stock_tickers: Union[List[str], str],
    ):
        if not isinstance(start_date, datetime.date) or not isinstance(
            end_date, datetime.date
        ):
            raise TypeError("start_date and end_date must be datetime.date objects")
        if start_date > end_date:
            raise ValueError("start_date should be less than or equal to end_date")
        if end_date < self.min_date or start_date > self.max_date:
            raise ValueError(
                f"start_date and end_date must be between {self.min_date} and {self.max_date}"
            )
        if isinstance(stock_tickers, str):
            if stock_tickers not in self.supported_tickers:
                raise ValueError(f"stock_ticker {stock_tickers} is not supported")
        elif isinstance(stock_tickers, Sequence):
            if not all(ticker in self.supported_tickers for ticker in stock_tickers):
                raise ValueError("One or more stock_tickers are not supported")
        else:
            raise TypeError(
                "stock_tickers must be either a string or a list of strings"
            )


if __name__ == "__main__":
    fs = FinanceStore(supported_tickers=[YEARLY_10_PERCENT])
    print(fs.df)
