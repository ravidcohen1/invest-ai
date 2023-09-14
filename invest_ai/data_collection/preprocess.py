# Load the uploaded FinanceStore class
from pathlib import Path
from typing import Tuple

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

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

    def prepare_datasets(self):
        print("Preparing train...")
        train_df = self.preprocess(
            start_date=self.cfg.time_frames.train.start_date,
            end_date=self.cfg.time_frames.train.end_date,
            train=True,
        )
        print("Preparing val...")
        val_df = self.preprocess(
            start_date=self.cfg.time_frames.val.start_date,
            end_date=self.cfg.time_frames.val.end_date,
            train=False,
        )
        print("Preparing test...")
        test_df = self.preprocess(
            start_date=self.cfg.time_frames.test.start_date,
            end_date=self.cfg.time_frames.test.end_date,
            train=False,
        )
        print("Binning...")
        train_df, val_df, test_df = self._target_binning(train_df, val_df, test_df)
        print(
            f"Done! Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}"
        )
        return train_df, val_df, test_df

    def preprocess(self, start_date, end_date, train: bool) -> pd.DataFrame:
        finance_df = self.finance_store.get_finance_for_dates(
            start_date=start_date,
            end_date=end_date,
            stock_tickers=list(self.cfg.stocks),
            melt=True,
        )
        if finance_df.empty:
            raise f"No finance data for dates {start_date} - {end_date}"
        news_df = self.news_store.get_news_for_dates(
            start_date=start_date,
            end_date=end_date,
            fetch_missing_dates=False,
            drop_articles=True,
        )
        if news_df.empty:
            raise f"No news data for dates {start_date} - {end_date}"

        print("Filling missing dates...")
        finance_df = self._fill_missing_dates(finance_df)
        print("Aggregating news...")
        news_df = self._process_and_agg_news(news_df)
        print("Merging finance and news...")
        df = pd.merge(finance_df, news_df, on="date", how="outer")
        print("Time windowing...")
        df = self._time_windowing(df)
        print("Feature engineering...")
        df = self._feature_engineering(df)
        df_x, df_y = self._xy_split(df)
        print("Computing returns...")
        df_y = self._compute_returns(df_x, df_y)
        print("Dropping features for last day...")
        df_x = self._drop_features_for_last_day(df_x)
        print("Scaling...")
        df_x = self._scaling(df_x)
        print("Finalizing...")
        df = self._finalize(df_x, df_y)
        return df

    def _finalize(self, df_x, df_y):
        """
        Finalizes the feature and target DataFrames for model training or evaluation.

        This function takes preprocessed features and targets, performs additional
        aggregation, and merges them into a single DataFrame. It specifically:

        1. Converts the 'date' column to datetime format if not already.
        2. Selects relevant features based on the configuration.
        3. Aggregates the features by 'ticker' and sample ID, converting them to lists.
        4. Merges the aggregated features with the target 'return' values.

        Parameters:
        -----------
        df_x : pd.DataFrame
            The DataFrame containing features. It is assumed to contain a 'date',
            'ticker', sample ID, and feature columns as specified in the configuration.

        df_y : pd.DataFrame
            The DataFrame containing the target 'return' values. It is assumed to
            contain a 'ticker', sample ID, and 'return' columns.

        Returns:
        --------
        pd.DataFrame
            The finalized DataFrame containing both features and target, ready for
            model training or evaluation.

        Assumptions:
        ------------
        - Both df_x and df_y are assumed to have 'ticker' and sample ID columns.
        - The 'date' column in df_x is assumed to be convertible to datetime format.
        """
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
        """
        Compute the returns based on the selling and buying metrics for each ticker and sample ID.

        This method computes returns based on the aggregated value of a selling metric in the
        horizon period (df_y) and the last value of a buying metric in the lookback period (df_x).

        Parameters
        ----------
        df_x : DataFrame
            The DataFrame containing data from the lookback period. It is assumed that this DataFrame
            contains the columns ["ticker", SAMPLE_ID] and the buying metric specified in the configuration.

        df_y : DataFrame
            The DataFrame containing data from the horizon period. It is assumed that this DataFrame
            contains the columns ["ticker", SAMPLE_ID] and the selling metric specified in the configuration.

        Returns
        -------
        DataFrame
            A new DataFrame with the same structure as df_y, but with an additional column 'return' that
            contains the computed returns.

        Assumptions
        -----------
        - df_x and df_y have the same tickers and SAMPLE_IDs.
        - df_x and df_y have non-overlapping DAY_IDX.
        - The buying_at and selling_at metrics specified in the configuration exist in df_x and df_y, respectively.
        - df_x and df_y are sorted by ["ticker", SAMPLE_ID, DAY_IDX] in ascending order.

        Notes
        -----
        - The aggregation method (mean, last, etc.) for the selling_at metric is configurable.
        - The buying_at metric's last value in the lookback period is used for computation.

        """
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
            for i in tqdm(range(len(ticker_data) - window_size + 1)):
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
        """
        Scale numerical features in the data according to a specified method and reference.

        This method scales the numerical features based on a scaling method and a reference feature.
        The scaling is performed in a way that's unique to each ['ticker', SAMPLE_ID] group.

        Parameters
        ----------
        df_x : DataFrame
            DataFrame containing features for the lookback period.
            It is assumed that this DataFrame contains the columns ["ticker", SAMPLE_ID] and the features specified in the configuration.

        df_y : DataFrame, optional
            DataFrame containing features for the horizon period.
            If provided, it is assumed that this DataFrame contains the columns ["ticker", SAMPLE_ID] and the features specified in the configuration.

        Returns
        -------
        DataFrame or Tuple[DataFrame, DataFrame]
            The scaled DataFrame(s). If `df_y` is provided, returns a tuple of scaled DataFrames `(df_x, df_y)`.

        Assumptions
        -----------
        - df_x and df_y, if provided, have the same tickers and SAMPLE_IDs.
        - The 'relative_to' feature exists in df_x and df_y, and is not among the features dropped for the last day.
        - df_x and df_y are sorted by ["ticker", SAMPLE_ID, DAY_IDX] in ascending order.

        Raises
        ------
        ValueError
            If 'relative_to' feature is among the features dropped for the last day.
        """

        def _scale_feature(df, feature, scale_value):
            scale_value = scale_value.copy()
            # Reset the index of scale_value to make it a DataFrame
            scale_value = scale_value.reset_index()
            scale_value = scale_value.rename(columns={feature: "scale_value"})

            # Merge this DataFrame with df to apply the scaling factor
            df = pd.merge(df, scale_value, on=["ticker", SAMPLE_ID], how="left")

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
                scale_value = df_x.groupby(["ticker", SAMPLE_ID])[feature].agg(
                    relative_to
                )

                # Apply the scaling to both df_x and df_y
                df_x = _scale_feature(df_x, feature, scale_value)
                if df_y is not None:
                    df_y = _scale_feature(df_y, feature, scale_value)
        else:
            raise NotImplemented(f"Not implemented scaling method {method}")
        if df_y is None:
            return df_x
        else:
            return df_x, df_y

    def _determine_bins(self, train_df: pd.DataFrame) -> pd.IntervalIndex:
        num_bins = self.cfg.target_binning.num_bins
        strategy = self.cfg.target_binning.strategy

        if strategy == "equal_frequency":
            _, bins = pd.qcut(train_df["return"], q=num_bins, retbins=True)
        elif strategy == "equal_width":
            _, bins = pd.cut(train_df["return"], bins=num_bins, retbins=True)
        else:
            raise ValueError(f"Unsupported binning strategy: {strategy}")

        return pd.IntervalIndex.from_breaks(bins)

    def _apply_bins(self, df: pd.DataFrame, bins: pd.IntervalIndex) -> pd.DataFrame:
        labels = self.cfg.target_binning.labels
        # Find the index of the interval for each "return" value
        bin_indices = bins.get_indexer(df["return"].values)

        # Map the index to the corresponding label
        df["target"] = [labels[int(i)] if i != -1 else None for i in bin_indices]

        return df

    def _target_binning(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Determine the bin edges using only the training data
        bins = self._determine_bins(train_df)

        # Apply the bin edges to the training, validation, and test data
        train_df = self._apply_bins(train_df, bins)
        if val_df is not None:
            val_df = self._apply_bins(val_df, bins)
        if test_df is not None:
            test_df = self._apply_bins(test_df, bins)

        return train_df, val_df, test_df


def visualise_this(cfg):
    # Initialize FinanceStore
    finance_store = FinanceStore(data_path=Path(cfg.finance_data_path))

    # Initialize NewsStore
    news_store = NewsStore(csv_file_path=Path(cfg.news_data_path))

    # Initialize DataPreprocessor with FinanceStore and preprocessing configurations
    cfg.data.target_binning.num_bins = 5
    cfg.data.target_binning.labels = ["very_bad", "bad", "neutral", "good", "very_good"]
    preprocessor = DataPreprocessor(finance_store, news_store, cfg.data)

    # Start data preprocessing
    import datetime

    start_date = datetime.date(year=2020, month=1, day=1)
    end_date = datetime.date(year=2020, month=12, day=30)
    df_train = preprocessor.preprocess(
        start_date=start_date, end_date=end_date, train=True
    )
    df_train, _, _ = preprocessor._target_binning(df_train)

    from invest_ai.plots.histograms import plot_return_histograms_by_target

    plot_return_histograms_by_target(df_train)
    print()


# @hydra.main(config_path="../configs", config_name="configs")
# def main(cfg: DictConfig) -> None:
#     visualise_this(cfg)
#
#
# if __name__ == "__main__":
#     main()
