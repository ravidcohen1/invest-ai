# Load the uploaded FinanceStore class
from pathlib import Path
from typing import Tuple

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from invest_ai.data_collection.finance_store import FinanceStore
from invest_ai.data_collection.news_store import NewsStore

SAMPLE_ID = "sample_id"
DAY_IDX = "day_idx"


class DataPreprocessor:
    """
    A class for preprocessing finance and news data according to specific configurations.
    """

    def __init__(
        self, finance_store: FinanceStore, news_store: NewsStore, config: DictConfig
    ):
        """
        Initialize the DataPreprocessor class.

        :param finance_store: An instance of the FinanceStore class.
        :type finance_store: FinanceStore
        :param news_store: An instance of the NewsStore class.
        :type news_store: NewsStore
        :param config: Configuration for data preprocessing.
        :type config: DictConfig
        """
        self.finance_store = finance_store
        self.news_store = news_store
        self.cfg = config

    def preprocess(
        self, start_date, end_date, train: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        finance_df = self.finance_store.get_finance_for_dates(
            start_date=start_date,
            end_date=end_date,
            stock_tickers=list(self.cfg.tickers),
            melt=True,
        )
        news_df = self.news_store.get_news_for_dates(
            start_date=start_date,
            end_date=end_date,
            fetch_missing_dates=False,
            drop_articles=True,
        )

        finance_df = self._fill_missing_dates(finance_df)
        news_df = self._process_and_agg_news(news_df)
        df = pd.merge(finance_df, news_df, on="date", how="outer")

        df = self._time_windowing(df)
        df = self._feature_engineering(df)
        df_x, df_y = self._xy_split(df)
        df_y = self._compute_returns(df_x, df_y)
        # df_y = self._target_binning(df_y)
        df_x = self._drop_features_for_last_day(df_x)
        df_x = self._scaling(df_x)

        df = self._finalize(df_x, df_y)
        return df

    def _finalize(self, df_x, df_y):
        # Convert the 'date' column to datetime format
        features = (
            self.cfg.features.numerical_features + self.cfg.features.textual_features
        )
        if "ticker" not in features:
            features.append("ticker")
        df_x = df_x[[SAMPLE_ID, "date"] + features]
        df_x_agg = df_x.groupby(["ticker", SAMPLE_ID]).agg(list).reset_index()
        df_y = df_y[["ticker", SAMPLE_ID, "return"]]

        # Merge aggregated df_x and df_y
        final_df = pd.merge(df_x_agg, df_y, on=["ticker", SAMPLE_ID], how="inner")

        return final_df

    def _process_and_agg_news(self, news_df):
        news_features = list(
            set(self.cfg.features.textual_features) & set(news_df.keys())
        )
        news_df = news_df.groupby("date")[news_features].agg(list).reset_index()
        news_df["date"] = pd.to_datetime(news_df["date"])
        return news_df

    def _drop_features_for_last_day(self, df_x: pd.DataFrame) -> pd.DataFrame:
        """
        Drop specified features for the last day of the lookback period to avoid lookahead bias.

        Parameters:
        df_x (pd.DataFrame): The features DataFrame

        Returns:
        pd.DataFrame: The features DataFrame with specified columns dropped for the last day of the lookback
        """
        features_to_drop = self.cfg.features.drop_features_for_last_day
        mask_last_day = df_x[DAY_IDX] == self.cfg.features.lookback - 1

        df_x.loc[mask_last_day, features_to_drop] = None
        return df_x

    def _compute_returns(self, df_x: pd.DataFrame, df_y: pd.DataFrame) -> pd.DataFrame:
        # Get the configurations
        selling_at = self.cfg.returns.selling_at
        aggregation = self.cfg.returns.aggregation
        buying_at = self.cfg.returns.buying_at

        # Compute the aggregated value for the selling_at metric in the horizon period
        df_y_aggregated = df_y.groupby(["ticker", SAMPLE_ID])[selling_at].aggregate(
            aggregation
        )

        # Get the last 'buying_at' metric value from the lookback period in df_x
        last_buying_at_value = df_x.groupby(["ticker", SAMPLE_ID])[buying_at].last()

        # Reset indices
        df_y_aggregated = df_y_aggregated.reset_index(drop=True)
        last_buying_at_value = last_buying_at_value.reset_index(drop=True)
        df_y = df_y.reset_index(drop=True)

        # Compute the returns
        df_y["return"] = (df_y_aggregated / last_buying_at_value) - 1

        return df_y

    def _fill_missing_dates(self, df):
        min_date = df["date"].min()
        max_date = df["date"].max()
        all_dates = pd.date_range(min_date, max_date, freq="D")

        # Step 2, 3, 4: Create complete DataFrame
        full_dfs = []

        for ticker in df["ticker"].unique():
            ticker_df = df[df["ticker"] == ticker]

            # Create DataFrame with all dates
            full_date_df = pd.DataFrame(all_dates, columns=["date"])
            full_date_df["ticker"] = ticker

            # Merge with existing data
            full_ticker_df = pd.merge(
                full_date_df, ticker_df, on=["date", "ticker"], how="left"
            )

            # Optional: Fill NaNs in other columns (customize as needed)
            # full_ticker_df['close'].fillna(method='ffill', inplace=True)

            full_dfs.append(full_ticker_df)

        # Concatenate all the DataFrames
        full_df = pd.concat(full_dfs).reset_index(drop=True)
        return full_df

    def _time_windowing(self, df: pd.DataFrame) -> pd.DataFrame:
        df.sort_values(by=["date"], inplace=True)

        lookback = self.cfg.features.lookback
        horizon = self.cfg.returns.horizon
        target_aggregation_method = self.cfg.returns.aggregation
        window_size = lookback + horizon

        windowed_data = []

        for ticker in df["ticker"].unique():
            ticker_data = df[df["ticker"] == ticker]
            for i in range(len(ticker_data) - window_size + 1):
                window = ticker_data.iloc[i : i + window_size].copy()
                if target_aggregation_method in ["first", "last"]:
                    target_data = window.iloc[-horizon:][self.cfg.returns.selling_at]
                    target_ids = 0 if target_aggregation_method == "first" else -1
                    aggregated_target = target_data.iloc[target_ids]
                    if pd.isnull(aggregated_target):
                        continue
                elif horizon <= 2:
                    raise NotImplemented("take care of this case")
                window[SAMPLE_ID] = i
                window[DAY_IDX] = range(window_size)
                windowed_data.append(window)

        df = pd.concat(windowed_data).reset_index(drop=True)
        return df

    def _xy_split(self, df):
        lookback = self.cfg.features.lookback
        x = df[df[DAY_IDX] < lookback]
        y = df[df[DAY_IDX] >= lookback]
        return x, y

    def _feature_engineering(self, df):
        df["weekday"] = df["date"].dt.day_name()
        return df

    def _scaling(self, df_x, df_y=None):
        def _scale_feature(df, feature, scale_value):
            # Create a DataFrame for the scale values, indexed by ['ticker', 's_id']
            scale_df = pd.DataFrame(
                {
                    "ticker": df["ticker"],
                    SAMPLE_ID: df[SAMPLE_ID],
                    "scale_value": scale_value,
                }
            ).drop_duplicates()

            # Merge this DataFrame with df_x and df_y to apply the scaling factor
            df = df.merge(scale_df, on=["ticker", SAMPLE_ID])

            # Apply the scaling
            df[feature] /= df["scale_value"]

            # Drop the temporary scale_value column
            df.drop(columns=["scale_value"], inplace=True)

            return df

        method = self.cfg.numerical_features_scaling.method
        relative_to = self.cfg.numerical_features_scaling.relative_to
        selected_features = self.cfg.features.numerical_features

        if relative_to in self.cfg.features.drop_features_for_last_day:
            raise ValueError(
                f"{relative_to} of the last day was dropped and cannot be a reference for scaling"
            )
        if method == "relative_scale":
            for feature in selected_features:
                # Calculate the scaling values only based on df_x
                scale_value = df_x.groupby(["ticker", SAMPLE_ID])[feature].transform(
                    relative_to
                )

                # Apply the scaling to both df_x and df_y
                df_x = _scale_feature(df_x, feature, scale_value)
                if df_y is not None:
                    df_y = _scale_feature(df_y, feature, scale_value)
        if df_y is None:
            return df_x
        else:
            return df_x, df_y

    def _target_binning(self, df_y: pd.DataFrame) -> pd.DataFrame:
        # Get the configurations
        num_bins = self.cfg.target_binning.num_bins
        labels = self.cfg.target_binning.labels
        strategy = self.cfg.target_binning.strategy

        # Perform the binning
        if strategy == "equal_frequency":
            df_y["target"] = pd.qcut(df_y["return"], q=num_bins, labels=labels)
        elif strategy == "equal_width":
            df_y["target"] = pd.cut(df_y["return"], bins=num_bins, labels=labels)
        else:
            raise ValueError(f"Unsupported binning strategy: {strategy}")

        return df_y


@hydra.main(config_path="../configs", config_name="configs")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Initialize FinanceStore
    finance_store = FinanceStore(data_path=Path(cfg.finance_data_path))

    # Initialize NewsStore
    news_store = NewsStore(csv_file_path=Path(cfg.news_data_path))

    # Initialize DataPreprocessor with FinanceStore and preprocessing configurations
    preprocessor = DataPreprocessor(finance_store, news_store, cfg.data)

    # Start data preprocessing
    import datetime

    start_date = datetime.date(year=2020, month=1, day=1)
    end_date = datetime.date(year=2020, month=12, day=30)
    df_x, df_y = preprocessor.preprocess(
        start_date=start_date, end_date=end_date, train=True
    )
    print()


if __name__ == "__main__":
    main()
