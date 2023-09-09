import os
from datetime import datetime
from pathlib import Path
from shutil import copy2

import pandas as pd
import pytest

from invest_ai.data_collection.news_store import NewsStore

TEST_DATA_PATH = Path("__file__").parent / "data/test_news_store.csv"


# Setup fixtures for reusable components
@pytest.fixture
def setup_news_store():
    ns = NewsStore(TEST_DATA_PATH, backup=False)
    if not TEST_DATA_PATH.exists():
        ns.get_news_for_dates("2021-09-01", "2021-09-02", fetch_missing_dates=True)
    return ns


# Test cases
def test_initialization_and_data_loading(setup_news_store):
    news_store = setup_news_store
    assert news_store is not None
    assert isinstance(news_store.df, pd.DataFrame)


def test_save_data(setup_news_store, tmp_path):
    news_store = setup_news_store
    test_csv = tmp_path / "test.csv"
    news_store.csv_file_path = test_csv
    news_store._save_data()
    assert test_csv.exists()


def test_backup_data(setup_news_store, tmp_path):
    news_store = setup_news_store
    news_store.csv_file_path = tmp_path / "existing.csv"
    copy2(TEST_DATA_PATH, news_store.csv_file_path)
    initial_files = set(os.listdir(tmp_path))
    news_store._backup_data()
    final_files = set(os.listdir(tmp_path))
    assert len(final_files) == len(initial_files) + 1
    new_files = final_files - initial_files
    assert any("news_store_backup_" in f for f in new_files)


import pytest

# ... other imports and fixtures ...


# Test cases for _validate_inputs method
@pytest.mark.parametrize(
    "start_date_str, end_date_str, should_raise",
    [
        ("2021-09-01", "2021-09-02", False),
        ("2021-09-02", "2021-09-01", True),
        ("invalid-date", "2021-09-02", True),
        ("2021-09-02", "invalid-date", True),
        ("invalid-date", "invalid-date", True),
    ],
)
def test_validate_inputs(setup_news_store, start_date_str, end_date_str, should_raise):
    if should_raise:
        with pytest.raises(Exception):
            setup_news_store._validate_inputs(start_date_str, end_date_str)
    else:
        setup_news_store._validate_inputs(start_date_str, end_date_str)


def test_get_news_for_dates(setup_news_store):
    start_date_str = "2021-09-01"
    end_date_str = "2021-09-02"

    # Fetch news data for the date range
    result_df = setup_news_store.get_news_for_dates(start_date_str, end_date_str)

    # Check if the result is a DataFrame
    assert isinstance(result_df, pd.DataFrame)

    # Check if the DataFrame is empty
    assert not result_df.empty

    # Check if the DataFrame has the expected columns
    expected_columns = ["date", "url", "source", "title", "article"]
    assert all(column in result_df.columns for column in expected_columns)

    # Check if the DataFrame contains news data for the specified date range
    result_dates = pd.to_datetime(result_df["date"]).dt.date
    assert all(
        start_date_str <= date <= end_date_str for date in result_dates.astype(str)
    )

    # Additional checks can include verifying the source, title, or article content, if needed.
