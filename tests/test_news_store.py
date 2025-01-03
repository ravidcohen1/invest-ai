import os
from datetime import datetime
from pathlib import Path
from shutil import copy2

import pandas as pd
import pytest

from invest_ai.data_collection.news_store import NewsStore


# Test cases
def test_initialization_and_data_loading(news_store):
    news_store = news_store
    assert news_store is not None
    assert isinstance(news_store.df, pd.DataFrame)


def test_save_data(news_store, tmp_path):
    news_store = news_store
    test_csv = tmp_path / "test.csv"
    news_store.csv_file_path = test_csv
    news_store._save_data()
    assert test_csv.exists()


def test_backup_data(news_store, tmp_path):
    news_store = news_store
    copy2(news_store.csv_file_path, tmp_path / "existing.csv")
    news_store.csv_file_path = tmp_path / "existing.csv"
    initial_files = set(os.listdir(tmp_path))
    news_store._backup_data()
    final_files = set(os.listdir(tmp_path))
    assert len(final_files) == len(initial_files) + 1
    new_files = final_files - initial_files
    assert any("news_store_backup_" in f for f in new_files)


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
def test_validate_inputs(news_store, start_date_str, end_date_str, should_raise):
    if should_raise:
        with pytest.raises(Exception):
            news_store._validate_inputs(start_date_str, end_date_str)
    else:
        news_store._validate_inputs(start_date_str, end_date_str)


def test_get_news_for_dates(news_store):
    start_date_str = "2023-08-01"
    end_date_str = "2023-08-30"

    # Fetch news data for the date range
    result_df = news_store.get_news_for_dates(start_date_str, end_date_str)

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


def test_get_news_for_dates_with_keywords(news_store):
    start_date_str = "2023-08-01"
    end_date_str = "2023-08-30"
    keywords = ["apple", "google", "microsoft"]
    news_store.keywords = keywords

    # Fetch news data for the date range with keywords
    result_df = news_store.get_news_for_dates(start_date_str, end_date_str)

    # Check if the result is a DataFrame
    assert isinstance(result_df, pd.DataFrame)

    # Check if the DataFrame is empty
    assert not result_df.empty

    # Check if the DataFrame has the expected columns, including "keywords_count"
    expected_columns = ["date", "url", "source", "title", "article", "keywords_count"]
    assert all(column in result_df.columns for column in expected_columns)

    # Check if the DataFrame contains news data for the specified date range
    result_dates = pd.to_datetime(result_df["date"]).dt.date
    assert all(
        start_date_str <= date <= end_date_str for date in result_dates.astype(str)
    )

    # Check if "keywords_count" is a non-negative integer
    assert all(
        isinstance(count, int) and count >= 0 for count in result_df["keywords_count"]
    )

    # Additional checks can include verifying the source, title, or article content, if needed.
