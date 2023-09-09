# Revised test cases for the FinanceStore class using pytest and mocking

import os
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from invest_ai.data_collection.finance_store import FinanceStore

TEST_DATA_PATH = Path("__file__").parent / "data/test_finance_store.pkl"


@pytest.fixture(scope="module")
def finance_store():
    fs = FinanceStore(
        min_date=date(2023, 8, 7),
        max_date=date(2023, 8, 12),
        supported_tickers=["AAPL", "GOOGL"],
        data_path=TEST_DATA_PATH,
    )
    return fs


# Test if yf.download is called when the file does not exist
@patch("yfinance.download")
def test_yf_download_called(mock_yf_download):
    mock_yf_download.return_value = pd.DataFrame()
    not_exist_path = "data/non_existent_file.pkl"
    min_date = date(2023, 8, 7)
    max_date = date(2023, 8, 12)
    supported_tickers = ["AAPL", "GOOGL"]
    if os.path.isfile(not_exist_path):
        os.remove(not_exist_path)

    # Case 1: File does not exist, yf.download should be called
    FinanceStore(
        min_date=min_date,
        max_date=max_date,
        supported_tickers=supported_tickers,
        data_path=Path(not_exist_path),
    )
    mock_yf_download.assert_called_with(
        tickers=supported_tickers, start=min_date, end=max_date, ignore_tz=True
    )

    # Case 2: File now exists, yf.download should not be called again
    mock_yf_download.reset_mock()  # Reset the mock to clear previous calls
    FinanceStore(
        min_date=min_date,
        max_date=max_date,
        supported_tickers=supported_tickers,
        data_path=Path(not_exist_path),
    )
    mock_yf_download.assert_not_called()


# Revised test case for get_finance_for_dates focusing on dates, keys, tickers, and NaN values

import numpy as np


def test_get_finance_for_dates(finance_store):
    result = finance_store.get_finance_for_dates(
        start_date=date(2023, 8, 7), end_date=date(2023, 8, 8), stock_tickers=["AAPL"]
    )

    # Check that the date range in the result is as expected
    assert result.index.min() == pd.Timestamp("2023-08-07")
    assert result.index.max() == pd.Timestamp("2023-08-08")

    # Check that only the specified tickers are in the columns
    assert "AAPL" in result.columns.get_level_values(1)
    assert "GOOGL" not in result.columns.get_level_values(1)

    # Check that the result DataFrame contains the expected keys (e.g., 'Adj Close', 'Volume')
    expected_keys = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for key in expected_keys:
        assert key in result.columns.get_level_values(0)

    # Check for not too many NaN values
    assert result.isna().sum().sum() == 0


from datetime import date
from pathlib import Path

import pytest


def test_finance_store_input_validation(finance_store):
    # Test invalid min_date and max_date types
    with pytest.raises(TypeError):
        FinanceStore(
            min_date="2023-08-07",
            max_date="2023-08-12",
            supported_tickers=["AAPL", "GOOGL"],
            data_path=Path("some_path"),
        )

    # Test min_date greater than max_date
    with pytest.raises(ValueError):
        FinanceStore(
            min_date=date(2023, 8, 13),
            max_date=date(2023, 8, 12),
            supported_tickers=["AAPL", "GOOGL"],
            data_path=Path("some_path"),
        )

    # Test invalid supported_tickers
    with pytest.raises(ValueError):
        FinanceStore(
            min_date=date(2023, 8, 7),
            max_date=date(2023, 8, 12),
            supported_tickers=[],
            data_path=Path("some_path"),
        )

    # Test invalid data_path type
    with pytest.raises(TypeError):
        FinanceStore(
            min_date=date(2023, 8, 7),
            max_date=date(2023, 8, 12),
            supported_tickers=["AAPL", "GOOGL"],
            data_path="some_path",
        )

    fs = finance_store
    # Test invalid start_date and end_date types in get_finance_for_dates
    with pytest.raises(TypeError):
        fs.get_finance_for_dates(
            start_date="2023-08-07", end_date=date(2023, 8, 9), stock_tickers=["AAPL"]
        )

    # Test start_date greater than end_date in get_finance_for_dates
    with pytest.raises(ValueError):
        fs.get_finance_for_dates(
            start_date=date(2023, 8, 9),
            end_date=date(2023, 8, 7),
            stock_tickers=["AAPL"],
        )

    # Test invalid stock_tickers in get_finance_for_dates
    with pytest.raises(ValueError):
        fs.get_finance_for_dates(
            start_date=date(2023, 8, 7),
            end_date=date(2023, 8, 9),
            stock_tickers="INVALID",
        )
