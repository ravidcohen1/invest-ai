import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd

from invest_ai.fine_tune.costs import compute_inference_costs, compute_training_costs
from invest_ai.fine_tune.fine_tuning import fine_tune_and_wait
from invest_ai.fine_tune.ft_data_preparation import (
    generate_prompt_and_completion_series,
    get_index_from_sample,
    prepare_sample_for_prompt_with_format,
)
from invest_ai.fine_tune.inference import create_completion
from invest_ai.return_predictor.baselines import AbstractReturnPredictor


class LLMReturnPredictor(AbstractReturnPredictor):
    """
    LLMReturnPredictor is responsible for using a Language Model for stock return prediction.
    It inherits from the AbstractReturnPredictor class and implements its abstract methods.
    """

    def __init__(
        self,
        model: str,
        epochs: Optional[int],
        caching_dir: str,
        max_budget: int,
        labels: List[str],
        tta: bool = False,
    ):
        self.model = model
        self.epochs = epochs
        self.caching_dir = Path(caching_dir)
        os.makedirs(self.caching_dir, exist_ok=True)
        self.max_budget = max_budget
        self.labels = labels
        self.model_id, self.logs = None, None
        self.features = None
        self.tta = tta

    def fit(
        self,
        train_df_path: Path,
        val_df_path: Optional[Path] = None,
        test_df_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        train_df_path = self._prepare_for_ft(train_df_path)
        val_df_path = self._prepare_for_ft(val_df_path)
        test_df_path = self._prepare_for_ft(test_df_path)

        estimated_cost = self._estimate_costs(train_df_path, val_df_path)
        if estimated_cost > self.max_budget:
            raise ValueError(
                f"Estimated cost ({estimated_cost}) exceeds the maximum budget ({self.max_budget})"
            )
        self.model_id, self.logs = fine_tune_and_wait(
            train_df_path,
            val_df_path,
            self.caching_dir,
            self.model,
            self.epochs,
            self.labels,
        )

        self.features = self._prepare_features(
            [train_df_path, val_df_path, test_df_path]
        )
        return self.logs

    def _prepare_features(self, df_paths: List[Path]) -> pd.DataFrame:
        features = []
        for df_path in df_paths:
            df = pd.read_csv(df_path)
            df = df[["index", "prompt"]]
            features.append(df)
        all_prompts = pd.concat(features).drop_duplicates().set_index("index")
        return all_prompts

    def predict(self, features: pd.Series) -> Any:
        """
        Predict the stock returns based on the given features.

        :param features: Features Series for prediction.
        :return: Predicted stock returns. The return type can vary based on the implementation.
        """
        sample_index = get_index_from_sample(features)
        if sample_index not in self.features.index:
            return None
        prompt = self.features.loc[sample_index].prompt
        if len(prompt) > 1 and isinstance(prompt, pd.Series):
            if self.tta:
                raise NotImplementedError("TTA not implemented for LLM.")
            else:
                prompt = prompt[0]

        label = create_completion(prompt, self.model_id)
        if label is not None:
            if label not in self.labels:
                print(f"Label {label} not in {self.labels}")
                return None
        return label

    def _prepare_for_ft(self, src_path: Path) -> Path:
        dst_path = self.caching_dir / (src_path.stem + ".csv")
        if dst_path.exists():
            return dst_path
        generate_prompt_and_completion_series(src_path, dst_path)
        return dst_path

    def _estimate_costs(self, train_path, val_path) -> float:
        """
        Estimate the costs for training, inference, and validation.

        Parameters:
            cfg: Configuration object containing various settings.
            train_path (Path): Path to the training data file.
            val_path (Path): Path to the validation data file.
            test_path (Path): Path to the test data file.

        Returns:
            float: The total cost.
        """
        print("Estimating costs...")
        costs_file_path = self.caching_dir / "costs.txt"
        if costs_file_path.exists():
            print("Costs already estimated. Skipping...")
            with open(costs_file_path, "r") as f:
                log_str = f.read()
            print(log_str)
            return float(log_str.split("\n")[-1].split(":")[-1].strip())
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        train_cost = compute_training_costs(
            train_df, epochs=self.epochs, model=self.model
        )
        train_inference_cost = compute_inference_costs(
            train_df.iloc[-len(val_df) :], model=self.model
        )
        val_cost = compute_inference_costs(val_df, model=self.model)

        total_cost = train_cost + train_inference_cost + val_cost
        log_str = (
            f"Training cost: {train_cost: .1f}\n"
            f"Training inference cost: {train_inference_cost: .1f}\n"
            f"Validation cost: {val_cost: .1f}\n"
            f"Total cost: {total_cost: .1f}"
        )
        print(log_str)
        with open(costs_file_path, "w") as f:
            f.write(log_str)
        return total_cost

    def get_labels(self) -> List[str]:
        return self.labels


if __name__ == "__main__":
    model = "babbage-002"
    epochs = 1
    caching_dir = "/Users/user/PycharmProjects/invest-ai/results/debug/fine_tuning/"
    max_budget = 5
    labels = ["very bad", "bad", "neutral", "good", "very good"]
    predictor = LLMReturnPredictor(model, epochs, caching_dir, max_budget, labels)
    train_path = Path(
        "/Users/user/PycharmProjects/invest-ai/results/debug/processed_data/train.pkl"
    )
    val_path = Path(
        "/Users/user/PycharmProjects/invest-ai/results/debug/processed_data/val.pkl"
    )
    test_path = Path(
        "/Users/user/PycharmProjects/invest-ai/results/debug/processed_data/test.pkl"
    )
    predictor.fit(train_path, val_path, test_path)
    print(predictor.predict(pd.read_pickle(test_path).iloc[0]))
