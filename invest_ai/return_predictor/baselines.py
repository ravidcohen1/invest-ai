import abc
import random
from typing import Any, List

import pandas as pd
from omegaconf import DictConfig


class AbstractReturnPredictor(abc.ABC):
    """
    Abstract base class for stock return predictors.
    Defines the interface that all return predictor implementations must follow.
    """

    @abc.abstractmethod
    def __init__(self, configs: DictConfig):
        """
        Initialize the AbstractReturnPredictor object.

        :param configs: Configuration settings for the predictor.
        :raises NotImplementedError: This method must be implemented by subclass.
        """
        self.configs = configs
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, features: pd.DataFrame, labels: pd.Series):
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

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame):
        """
        Fit the return predictor model to the given data.

        Note: This implementation does not actually fit a model, and this method is a placeholder.

        :param features: Features DataFrame for training.
        :param labels: Labels DataFrame for training.
        """
        pass

    def predict(self, features: pd.DataFrame) -> str:
        """
        Predict a random label based on the given features.

        :param features: Features DataFrame for prediction.
        :return: A randomly selected label from the predefined list.
        """
        return random.choice(self.labels)
