import datetime
import os
from pathlib import Path
from typing import List, Union

import pandas as pd
import yfinance as yf

from invest_ai import configs as cfg


class FinanceStore:
    """
    A class to manage the storage and retrieval of finance data and their metadata.
    """

    def __init__(
        self,
        min_date: datetime.date = cfg.MIN_SUPPORTED_DATE,
        max_date: datetime.date = cfg.MAX_SUPPORTED_DATE,
        supported_tickers: List[str] = cfg.SUPPORTED_TICKERS,
        data_path: Path = cfg.FINANCE_DATA_PATH,
    ):
        self._validate_init_input(min_date, max_date, supported_tickers, data_path)
        self.data_path = data_path
        self.min_date = min_date
        self.max_date = max_date
        self.supported_tickers = supported_tickers
        self.df = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """
        Load finance data from the PKL file into a DataFrame.

        :return: DataFrame containing finance data.
        :rtype: pd.DataFrame
        """
        if os.path.exists(self.data_path):
            return pd.read_pickle(self.data_path)
        else:
            data = self._download_stock_prices(
                tickers=self.supported_tickers,
                start_date=self.min_date,
                end_date=self.max_date,
            )
            os.makedirs(self.data_path.parent, exist_ok=True)
            data.to_pickle(self.data_path)
            return data

    @staticmethod
    def _download_stock_prices(
        tickers: List[str], start_date: datetime.date, end_date: datetime.date
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
        stock_tickers: Union[List[str], str],
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
        sliced_df = self.df.loc[pd.IndexSlice[start_date:end_date], :]

        # If stock_tickers is a string, convert to a single-item list
        if isinstance(stock_tickers, str):
            stock_tickers = [stock_tickers]

        # Filter by stock_tickers
        sliced_df = sliced_df.loc[:, pd.IndexSlice[:, stock_tickers]]

        return sliced_df

    def _validate_init_input(
        self,
        min_date: datetime.date,
        max_date: datetime.date,
        supported_tickers: List[str],
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
        if isinstance(stock_tickers, str):
            if stock_tickers not in self.supported_tickers:
                raise ValueError(f"stock_ticker {stock_tickers} is not supported")
        elif isinstance(stock_tickers, list):
            if not all(ticker in self.supported_tickers for ticker in stock_tickers):
                raise ValueError("One or more stock_tickers are not supported")
        else:
            raise TypeError(
                "stock_tickers must be either a string or a list of strings"
            )


if __name__ == "__main__":
    fs = FinanceStore()
    print(fs.df)
