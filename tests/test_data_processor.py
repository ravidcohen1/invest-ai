import os
from datetime import date, datetime
from pathlib import Path
from shutil import copy2

import pandas as pd
import pytest
from omegaconf import OmegaConf

from invest_ai.data_collection.finance_store import FinanceStore
from invest_ai.data_collection.news_store import NewsStore
from invest_ai.data_collection.preprocess import DataPreprocessor

NEWS_DATA_PATH = Path("__file__").parent / "data/test_news_store.csv"
FINANCE_DATA_PATH = Path("__file__").parent / "data/test_finance_store.pkl"


# Setup fixtures for reusable components
@pytest.fixture
def setup_news_store():
    ns = NewsStore(NEWS_DATA_PATH, backup=False)
    return ns


@pytest.fixture(scope="module")
def finance_store():
    fs = FinanceStore(data_path=FINANCE_DATA_PATH)
    return fs


@pytest.fixture
def setup_data_preprocessor(setup_news_store, finance_store):
    config = OmegaConf.load("test_data_configs.yaml")
    return DataPreprocessor(finance_store, setup_news_store, config)


def test_prepare_datasets(setup_data_preprocessor):
    dp = setup_data_preprocessor
    # Your assertions here
    assert 1 == 1  # Replace with real tests


def test_fill_missing_dates(setup_data_preprocessor):
    dp = setup_data_preprocessor

    # Scenario 1: Basic functionality
    input_df1 = pd.DataFrame(
        {
            "date": [datetime(2021, 9, 1), datetime(2021, 9, 3)],
            "ticker": ["AAPL", "AAPL"],
            "close": [150, 152],
        }
    )
    output_df1 = dp._fill_missing_dates(input_df1)
    expected_df1 = pd.DataFrame(
        {
            "date": [datetime(2021, 9, 1), datetime(2021, 9, 2), datetime(2021, 9, 3)],
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "close": [150, None, 152],
        }
    )
    pd.testing.assert_frame_equal(output_df1, expected_df1, check_like=True)

    # Scenario 2: No missing dates
    input_df2 = pd.DataFrame(
        {
            "date": [datetime(2021, 9, 1), datetime(2021, 9, 2)],
            "ticker": ["AAPL", "AAPL"],
            "close": [150, 151],
        }
    )
    output_df2 = dp._fill_missing_dates(input_df2)
    pd.testing.assert_frame_equal(output_df2, input_df2, check_like=True)

    # Scenario 3: Multiple tickers
    input_df3 = pd.DataFrame(
        {
            "date": [datetime(2021, 9, 1), datetime(2021, 9, 3), datetime(2021, 9, 1)],
            "ticker": ["AAPL", "AAPL", "GOOGL"],
            "close": [150, 152, 2800],
        }
    )
    output_df3 = dp._fill_missing_dates(input_df3)
    expected_df3 = pd.DataFrame(
        {
            "date": [
                datetime(2021, 9, 1),
                datetime(2021, 9, 2),
                datetime(2021, 9, 3),
                datetime(2021, 9, 1),
                datetime(2021, 9, 2),
                datetime(2021, 9, 3),
            ],
            "ticker": ["AAPL", "AAPL", "AAPL", "GOOGL", "GOOGL", "GOOGL"],
            "close": [150, None, 152, 2800, None, None],
        }
    )
    pd.testing.assert_frame_equal(output_df3, expected_df3, check_like=True)


def test__process_and_agg_news(setup_data_preprocessor):
    dp = setup_data_preprocessor

    # Scenario 1 and 3: Basic functionality and Multiple news features
    input_df1 = pd.DataFrame(
        {
            "date": [datetime(2021, 9, 1), datetime(2021, 9, 1), datetime(2021, 9, 2)],
            "feature1": ["news1", "news2", "news3"],
            "feature2": ["other1", "other2", "other3"],
        }
    )

    # Assuming the config specifies 'feature1' and 'feature2' as textual_features
    dp.cfg.features.textual_features = ["feature1", "feature2"]

    output_df1 = dp._process_and_agg_news(input_df1)

    expected_df1 = pd.DataFrame(
        {
            "date": [datetime(2021, 9, 1), datetime(2021, 9, 2)],
            "feature1": [["news1", "news2"], ["news3"]],
            "feature2": [["other1", "other2"], ["other3"]],
        }
    )

    # Convert 'date' to datetime format in the expected DataFrame
    expected_df1["date"] = pd.to_datetime(expected_df1["date"])

    # Scenario 2: Date conversion
    assert output_df1["date"].dtype == "datetime64[ns]"

    # Final DataFrame equality check
    pd.testing.assert_frame_equal(output_df1, expected_df1, check_like=True)


import pytest


# Parameterize the test to run through different configurations
@pytest.mark.parametrize(
    "lookback,horizon,aggregation,selling_at",
    [
        (5, 1, "last", "open"),
        (5, 1, "first", "close"),
        (5, 2, "last", "close"),
    ],
)
def test__time_windowing(
    setup_data_preprocessor, lookback, horizon, aggregation, selling_at
):
    dp = setup_data_preprocessor

    # Set up config according to parameters
    dp.cfg.features.lookback = lookback
    dp.cfg.returns.horizon = horizon
    dp.cfg.returns.aggregation = aggregation
    dp.cfg.returns.selling_at = selling_at

    # Prepare sample DataFrame (customize as needed)
    input_df = pd.DataFrame(
        {
            "date": list(pd.date_range(start="2021-01-01", end="2021-01-10", freq="D"))
            + list(pd.date_range(start="2021-01-01", end="2021-01-10", freq="D")),
            "ticker": ["AAPL"] * 10 + ["META"] * 10,
            "open": list(range(90, 100)) + list(range(100, 110)),
            "close": list(range(135, 145)) + list(range(105, 115)),
        }
    )

    # Run the function
    output_df = dp._time_windowing(input_df.copy())

    # Assertion 1: Check the number of unique windows
    unique_windows = len(output_df[["sample_id", "ticker"]].drop_duplicates())
    expected_windows = 2 * (int(len(input_df) / 2) - lookback - horizon + 1)
    assert (
        unique_windows == expected_windows
    ), f"Expected {expected_windows} unique windows, got {unique_windows}"

    # Assertion 2: Check DAY_IDX range
    day_idx_range = output_df["day_idx"].unique()
    expected_day_idx_range = list(range(lookback + horizon))
    assert set(day_idx_range) == set(
        expected_day_idx_range
    ), f"Expected DAY_IDX range {expected_day_idx_range}, got {day_idx_range}"

    # Assertion 3: Check aggregated target if applicable for all windows
    for ticker in output_df["ticker"].unique():
        for window_id in range(unique_windows // 2):
            current_window_data = output_df[
                (output_df["sample_id"] == window_id) & (output_df["ticker"] == ticker)
            ]

            assert (
                len(current_window_data) == horizon + lookback
            ), f"For window {window_id}, expected {horizon + lookback} rows, got {len(current_window_data)}"
            for metric in ["open", "close"]:
                (
                    current_window_data[metric]
                    == input_df[input_df.ticker == ticker][metric]
                    .iloc[window_id : window_id + horizon + lookback]
                    .values.tolist()
                ).all(), f"Expected values missmatch for {metric} in window {window_id} for ticker {ticker}"

    # check that Null values are being handled
    input_df[selling_at].iloc[-1] = None
    output_df = dp._time_windowing(input_df.copy())
    unique_windows = len(output_df[["sample_id", "ticker"]].drop_duplicates())
    expected_windows = 2 * (int(len(input_df) / 2) - lookback - horizon + 1) - 1
    assert (
        unique_windows == expected_windows
    ), f"Expected {expected_windows} unique windows, got {unique_windows}"

    output_df.groupby(["ticker", "sample_id"])[selling_at].agg(
        aggregation
    ).notnull().all(), f"Null values not being handled correctly for {selling_at} aggregation {aggregation}, horizon {horizon}, lookback {lookback}"


def test__feature_engineering(setup_data_preprocessor):
    dp = setup_data_preprocessor

    # Generate test DataFrame
    sample_data = {
        "date": pd.date_range(start="2022-01-01", end="2022-01-07"),
        "ticker": ["AAPL"] * 7,
        "open": range(1, 8),
    }
    test_df = pd.DataFrame(sample_data)

    # Apply the feature engineering method
    engineered_df = dp._feature_engineering(test_df.copy())

    # Validate the engineered DataFrame
    # 1. Check if the 'weekday' column has been added
    assert "weekday" in engineered_df.columns, "The 'weekday' column is missing."

    # 2. Check if the 'weekday' column contains the correct day names
    expected_weekdays = [
        "Saturday",
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
    ]
    assert all(
        engineered_df["weekday"] == expected_weekdays
    ), "The 'weekday' column contains incorrect day names."


def test__xy_split(setup_data_preprocessor):
    dp = setup_data_preprocessor

    # Generate test DataFrame with DAY_IDX column
    sample_data = {
        "day_idx": list(range(10)),
        "ticker": ["AAPL"] * 10,
        "open": range(1, 11),
    }
    test_df = pd.DataFrame(sample_data)

    # Apply the _xy_split method
    x, y = dp._xy_split(test_df.copy())

    # Validate the X and Y DataFrames
    # 1. Check if X contains rows with DAY_IDX < lookback
    assert all(
        x["day_idx"] < dp.cfg["features"]["lookback"]
    ), "X contains rows with DAY_IDX >= lookback."

    # 2. Check if Y contains rows with DAY_IDX >= lookback
    assert all(
        y["day_idx"] >= dp.cfg["features"]["lookback"]
    ), "Y contains rows with DAY_IDX < lookback."


# Test for _compute_returns method
def test__compute_returns(setup_data_preprocessor):
    dp = setup_data_preprocessor

    # Generate test DataFrame for df_x and df_y
    sample_data_x = {
        "day_idx": list(range(5)) * 2,
        "ticker": ["AAPL"] * 5 + ["GOOGL"] * 5,
        "sample_id": [1] * 5 + [2] * 5,
        "open": list(range(1, 6)) * 2,
    }
    df_x = pd.DataFrame(sample_data_x)

    sample_data_y = {
        "day_idx": list(range(5, 6)) * 2,
        "ticker": ["AAPL", "GOOGL"],
        "sample_id": [1, 2],
        "open": [6, 7],
    }
    df_y = pd.DataFrame(sample_data_y)

    # Apply the _compute_returns method
    computed_df = dp._compute_returns(df_x.copy(), df_y.copy())

    # Validate the computed DataFrame
    # 1. Check if the 'return' column has been added
    assert "return" in computed_df.columns, "The 'return' column is missing."

    # 2. Manually compute the expected returns and compare
    expected_return_aapl = (6 / 5) - 1
    expected_return_googl = (7 / 5) - 1
    assert (
        computed_df.loc[computed_df["ticker"] == "AAPL", "return"].iloc[0]
        == expected_return_aapl
    ), "Incorrect return for AAPL."
    assert (
        computed_df.loc[computed_df["ticker"] == "GOOGL", "return"].iloc[0]
        == expected_return_googl
    ), "Incorrect return for GOOGL."


# Test for _drop_features_for_last_day method
def test__drop_features_for_last_day(setup_data_preprocessor):
    dp = setup_data_preprocessor

    # Generate test DataFrame for df_x
    sample_data_x = {
        "day_idx": list(range(5)) * 2,
        "ticker": ["AAPL"] * 5 + ["GOOGL"] * 5,
        "sample_id": [1] * 5 + [2] * 5,
        "open": list(range(1, 6)) * 2,
        "close": list(range(6, 11)) * 2,
    }
    df_x = pd.DataFrame(sample_data_x)

    # Apply the _drop_features_for_last_day method
    modified_df_x = dp._drop_features_for_last_day(df_x.copy())

    # Validate the modified DataFrame
    # 1. Check if specified features are dropped for the last day
    last_day_mask = modified_df_x["day_idx"] == (dp.cfg["features"]["lookback"] - 1)
    assert (
        modified_df_x.loc[last_day_mask, "open"].notna().all()
    ), "Feature 'open' was dropped for the last day."
    assert (
        modified_df_x.loc[last_day_mask, "close"].isna().all()
    ), "Feature 'close' was not dropped for the last day."


from typing import Union

import numpy as np
import pandas as pd


def get_aggregated_value(df: pd.Series, method: str) -> Union[float, pd.Series]:
    if method == "first":
        return df.iloc[0]
    elif method == "last":
        return df.iloc[-1]
    elif method == "min":
        return df.min()
    elif method == "max":
        return df.max()
    elif method == "mean":
        return df.mean()
    else:
        raise ValueError(
            f"Invalid method: {method}. Choose from ['first', 'last', 'min', 'max', 'mean']."
        )


@pytest.mark.parametrize(
    "method,relative_to",
    [
        ("relative_scale", "first"),
        ("relative_scale", "last"),
        ("relative_scale", "min"),
        ("relative_scale", "max"),
        ("relative_scale", "mean"),
    ],
)
def test__scaling(setup_data_preprocessor, method, relative_to):
    dp = setup_data_preprocessor

    # Overwrite configurations for this test
    dp.cfg["features"]["numerical_features"] = ["open", "close"]
    dp.cfg["numerical_features_scaling"]["method"] = method
    dp.cfg["numerical_features_scaling"]["relative_to"] = relative_to

    # Generate test DataFrame for df_x and df_y
    sample_data_x = {
        "day_idx": list(range(5)) * 3,
        "ticker": ["AAPL"] * 5 + ["GOOGL"] * 5 + ["GOOGL"] * 5,
        "sample_id": [1] * 5 + [1] * 5 + [2] * 5,
        "open": list(range(1, 6)) + list(range(3, 8)) + [1, 3, 5, 7, 4],
        "close": list(range(6, 11)) + list(range(8, 13)) + [9, 4, 2, 3, 4],
    }
    df_x = pd.DataFrame(sample_data_x)

    sample_data_y = {
        "day_idx": list(range(5, 6)) * 3,
        "ticker": ["AAPL", "GOOGL", "GOOGL"],
        "sample_id": [1, 1, 2],
        "open": [6, 7, 8],
        "close": [11, 12, 13],
    }
    df_y = pd.DataFrame(sample_data_y)

    # Apply the _scaling method
    scaled_df_x, scaled_df_y = dp._scaling(df_x.copy(), df_y.copy())
    # Validate the scaled DataFrame

    for t in ["AAPL", "GOOGL"]:
        for s in [1, 2]:
            for m in ["open", "close"]:
                x_data = df_x[(df_x.ticker == t) & (df_x.sample_id == s)].sort_values(
                    "day_idx"
                )[m]
                if x_data.empty:
                    continue
                scale_val = get_aggregated_value(x_data, relative_to)

                out_x = scaled_df_x.loc[
                    (scaled_df_x["sample_id"] == s) & (scaled_df_x.ticker == t), m
                ]
                expected_x = (
                    df_x.loc[(df_x["sample_id"] == s) & (df_x.ticker == t), m]
                    / scale_val
                )
                assert np.allclose(
                    out_x, expected_x
                ), f"Incorrect scaling in x. t {t}, m {m}, s {s}, relative_to {relative_to}"
                out_y = scaled_df_y.loc[
                    (scaled_df_y["sample_id"] == s) & (scaled_df_y.ticker == t), m
                ]
                expected_y = (
                    df_y.loc[(df_y["sample_id"] == s) & (df_y.ticker == t), m]
                    / scale_val
                )
                assert np.allclose(
                    out_y, expected_y
                ), f"Incorrect scaling in y. t {t}, m {m}, s {s}, relative_to {relative_to}"


@pytest.mark.parametrize(
    "numerical_features, textual_features",
    [(["open", "close"], ["title"]), (["volume", "open"], ["weekday"])],
)
def test__finalize(setup_data_preprocessor, numerical_features, textual_features):
    dp = setup_data_preprocessor

    # Overwrite configurations for this test
    dp.cfg["features"]["numerical_features"] = numerical_features
    dp.cfg["features"]["textual_features"] = textual_features

    # Generate test DataFrame for df_x
    sample_data_x = {
        "sample_id": [1, 1, 1, 1, 1, 1, 2, 2, 2],
        "date": pd.date_range(start="2022-01-01", periods=3).tolist() * 2
        + pd.date_range(start="2022-01-04", periods=3).tolist(),
        "ticker": ["AAPL"] * 3 + ["GOOGL"] * 6,
        "open": [1, 2, 3, 4, 5, 6, 2, 3, 4],
        "close": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
        "volume": [100, 200, 300, 400, 500, 600, 20, 30, 40],
        "title": ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"],
        "weekday": ["Mon", "Tue", "Wed", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
    }
    df_x = pd.DataFrame(sample_data_x)

    # Generate test DataFrame for df_y
    sample_data_y = {
        "sample_id": [1, 2],
        "ticker": ["AAPL", "GOOGL"],
        "return": [0.1, 0.2],
    }
    df_y = pd.DataFrame(sample_data_y)

    # Apply the _finalize method
    final_df = dp._finalize(df_x, df_y)

    # Validate the final DataFrame
    # 1. Check if it contains the right columns
    expected_columns = (
        ["ticker", "sample_id"]
        + numerical_features
        + textual_features
        + ["return", "date"]
    )
    assert set(final_df.columns) == set(
        expected_columns
    ), f"Columns do not match. Got {final_df.columns}, expected {expected_columns}"

    # 2. Check if the data is aggregated correctly
    for feature in numerical_features + textual_features:
        assert all(
            isinstance(item, list) for item in final_df[feature]
        ), f"Feature {feature} is not aggregated into lists."

    # 3. Check if the 'return' values are correctly merged
    assert all(
        final_df["return"] == df_y.set_index(["ticker", "sample_id"])["return"].values
    ), "Return values are not correctly merged."


def validate_weekdays_make_sense(list_of_days):
    expected_days = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ]

    # Map each day to its corresponding index in a week (0-indexed)
    day_to_index = {day: index for index, day in enumerate(expected_days)}

    # Convert the list of days to list of indices
    indices = [day_to_index[day] for day in list_of_days]

    # Check if indices are in ascending order considering the wrap-around
    is_ascending = all(
        (indices[i + 1] - indices[i]) % 7 == 1 for i in range(len(indices) - 1)
    )

    assert (
        is_ascending
    ), f"The weekdays are not in the correct consecutive order. Got {list_of_days}"


def test_preprocess_end_to_end(
    setup_data_preprocessor, setup_news_store, finance_store
):
    dp = setup_data_preprocessor

    # Define the test parameters
    start_date = date(2023, month=8, day=1)
    end_date = date(2023, month=8, day=20)
    train = True

    # Run the preprocess function
    final_df = dp.preprocess(start_date, end_date, train)

    # Validate the final DataFrame
    # 1. Check if it contains the right columns based on the configuration
    expected_features = (
        dp.cfg.features.numerical_features + dp.cfg.features.textual_features
    )
    expected_columns = ["ticker", "sample_id"] + expected_features + ["return", "date"]
    assert set(final_df.columns) == set(
        expected_columns
    ), f"Columns do not match. Got {final_df.columns}, expected {expected_columns}"

    # 2. Check if the data is aggregated correctly
    for feature in expected_features:
        if feature not in ["ticker"]:
            assert all(
                isinstance(item, list) for item in final_df[feature]
            ), f"Feature {feature} is not aggregated into lists."
    assert all(
        isinstance(item, str) for item in final_df["ticker"]
    ), f"Feature {feature} is not a string"

    # 3. Check if 'return' values exist and are numeric
    assert pd.api.types.is_numeric_dtype(
        final_df["return"]
    ), "'return' column is not numeric."

    #  check some random examples
    for _, sample in final_df.sample(10).iterrows():
        for m in dp.cfg.features.numerical_features:
            assert m in sample
            if m in dp.cfg.features.drop_features_for_last_day:
                assert pd.isnull(sample[m][-1]), f"{m} is not null"
            else:
                assert pd.notnull(sample[m][-1]) or sample["weekday"][-1] in [
                    "Saturday",
                    "Sunday",
                ], f"{m} is null"
        assert isinstance(sample["title"], list)
        assert isinstance(sample["title"][0], list)
        assert isinstance(sample["title"][0][0], str)

        ticker = sample["ticker"]
        dates = sample["date"]
        assert (
            dp.cfg.returns.horizon == 1
        ), "I was counting on dp.cfg.returns.horizon==1"
        from datetime import timedelta

        start_date = dates[0]
        end_date = dates[-1] + timedelta(days=1)
        news_df = setup_news_store.get_news_for_dates(
            start_date, end_date, drop_articles=True
        )
        titles = news_df.groupby("date")["title"].agg(list)[:-1]
        assert (sample.title == titles).all(), "Titles are not right"

        validate_weekdays_make_sense(sample["weekday"])

        finance_df = finance_store.get_finance_for_dates(
            start_date=start_date, end_date=end_date, stock_tickers=[ticker], melt=True
        )
        y = finance_df[dp.cfg.returns.selling_at].iloc[-1]
        x = finance_df[dp.cfg.returns.buying_at].iloc[-2]
        assert np.isclose(sample["return"], y / x - 1)


# def test_binning():
#     assert False
