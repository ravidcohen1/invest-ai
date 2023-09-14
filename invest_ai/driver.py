import datetime
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
from invest_ai.data_collection.finance_store import FinanceStore
from invest_ai.data_collection.news_store import NewsStore
from invest_ai.data_collection.preprocess import DataPreprocessor
from invest_ai.investor.abstract_investor import AbstractInvestor
from invest_ai.simulation.bank import Bank
from invest_ai.simulation.simulator import Simulator
from invest_ai.utils.string import str_to_date


def run_simulation(cfg, subset="test") -> None:
    if subset == "train":
        raise NotImplementedError(
            "Training simulation not implemented... Take care of the dates"
        )
    start_date = cfg.experiment.time_frames[subset].start_date
    end_date = cfg.experiment.time_frames[subset].end_date
    fs = FinanceStore(
        min_date=start_date,
        max_date=end_date,
        supported_tickers=cfg.experiment.stocks,
        data_path=cfg.finance_data_path,
    )
    bank = Bank(
        start_date=start_date,
        fs=fs,
        initial_amount=cfg.experiment.budget.initial_amount,
    )
    investor: AbstractInvestor = hydra.utils.instantiate(cfg.experiment.investor)
    simulator = Simulator(
        bank=bank,
        investor=investor,
        monthly_budget=cfg.experiment.budget.monthly_budget,
    )
    simulation_days = (end_date - start_date).days
    for i in range(simulation_days):
        status = simulator.next_day()
        stats = status.to_dict()
        stats = {f"{k} - {subset}": v for k, v in stats.items()}
        wandb.log(stats)


def run_simulations(cfg):
    wandb.init(
        project="invest-ai",
        name=cfg.experiment.experiment_name,
        config=OmegaConf.to_container(cfg),
    )
    run_simulation(cfg, subset="train")
    run_simulation(cfg, subset="val")
    run_simulation(cfg, subset="test")


def prepare_data(cfg: DictConfig) -> None:
    start_date = cfg.experiment.time_frames["train"].start_date
    end_date = cfg.experiment.time_frames["test"].end_date
    fs = FinanceStore(
        min_date=start_date,
        max_date=end_date,
        supported_tickers=list(cfg.experiment.stocks),
        data_path=Path(cfg.finance_data_path),
    )
    news_store = NewsStore(csv_file_path=cfg.news_data_path, backup=False)
    dp = DataPreprocessor(fs, news_store, cfg.experiment)
    train_df, val_df, test_df = dp.prepare_datasets()
    os.makedirs(cfg.processed_data_path, exist_ok=True)
    train_df.to_pickle(Path(cfg.processed_data_path) / "train.pkl")
    val_df.to_pickle(Path(cfg.processed_data_path) / "val.pkl")
    test_df.to_pickle(Path(cfg.processed_data_path) / "test.pkl")


@hydra.main(config_path="configs", config_name="configs")
def main(cfg: DictConfig) -> None:
    prepare_data(cfg)

    # run_simulations()


if __name__ == "__main__":
    main()
