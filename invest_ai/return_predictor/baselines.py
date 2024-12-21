import abc
import random
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from omegaconf import DictConfig


class AbstractReturnPredictor(abc.ABC):
    """
    Abstract base class for stock return predictors.
    Defines the interface that all return predictor implementations must follow.
    """

    @abc.abstractmethod
    def fit(
        self, train_df_path: Path, val_df_path: Optional[Path] = None, **kwargs
    ) -> pd.DataFrame:
        """
        Fit the return predictor model to the given data.

        :param features: Features DataFrame for training.
        :param labels: Labels Series for training.
        :raises NotImplementedError: This method must be implemented by subclass.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, features: pd.Series) -> Any:
        """
        Predict the returns based on the given features.

        :param features: Features Series for prediction.
        :return: Predicted returns. The return type can vary based on the implementation.
        :raises NotImplementedError: This method must be implemented by subclass.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_labels(self) -> List[str]:
        """
        Get the list of labels from the given DataFrame.

        :return: List of labels.
        """
        raise NotImplementedError


class RandomReturnPredictor(AbstractReturnPredictor):
    """
    A return predictor that randomly selects a label from a predefined list.

    This class ignores the features and labels used for fitting, and simply
    returns a random label for each prediction.
    """

    def __init__(self, labels: List[str]):
        """
        Initialize the RandomReturnPredictor object.

        :param labels: List of labels to randomly select from during prediction.
        """
        self.labels = labels

    def fit(
        self, train_df_path: Path, val_df_path: Optional[Path] = None
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def predict(self, features: pd.DataFrame) -> str:
        """
        Predict a random label based on the given features.

        :param features: Features DataFrame for prediction.
        :return: A randomly selected label from the predefined list.
        """
        return random.choice(self.labels)

    def get_labels(self) -> List[str]:
        return self.labels


from invest_ai.fine_tune.ft_data_preparation import (
    generate_prompt_and_completion_series,
    get_index_from_sample,
    prepare_sample_for_prompt_with_format,
)


class GroundTrueReturnPredictor(AbstractReturnPredictor):
    """
    A return predictor that returns the ground truth label from the given DataFrame.
    """

    def __init__(self, labels: List[str]):
        """
        Initialize the RandomReturnPredictor object.

        :param labels: List of labels to randomly select from during prediction.
        """
        self.labels = labels
        self.ground_truth = None

    def fit(
        self,
        train_df_path: Path,
        val_df_path: Optional[Path] = None,
        test_df_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        train_labels = self._prepare_labels(train_df_path)
        val_labels = self._prepare_labels(val_df_path)
        test_labels = self._prepare_labels(test_df_path)
        self.ground_truth = (
            pd.concat([train_labels, val_labels, test_labels])[["index", "label"]]
            .drop_duplicates()
            .set_index("index")["label"]
        )
        return pd.DataFrame()

    def _prepare_labels(self, df_path: Path) -> pd.Series:
        def compute_label(return_value):
            if return_value > 0:
                return self.labels[-1]
            elif return_value < 0:
                return self.labels[0]
            else:
                return self.labels[len(self.labels) // 2]

        df = pd.read_pickle(df_path)
        df["index"] = df.apply(get_index_from_sample, axis=1)
        df["label"] = df["return"].apply(compute_label)
        return df

    def predict(self, features: pd.Series) -> str:
        sample_index = get_index_from_sample(features)
        if sample_index not in self.ground_truth.index:
            return None
        label = self.ground_truth[sample_index]
        assert label in self.labels, f"Label {label} not in {self.labels}"
        return label

    def get_labels(self) -> List[str]:
        return self.labels
