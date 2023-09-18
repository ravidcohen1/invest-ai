import datetime
import os
from pathlib import Path
from typing import Tuple

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb
from invest_ai.data_collection.finance_store import FinanceStore
from invest_ai.data_collection.news_store import NewsStore
from invest_ai.data_collection.preprocess import DataPreprocessor
from invest_ai.investor.abstract_investor import AbstractInvestor
from invest_ai.return_predictor.baselines import AbstractReturnPredictor
from invest_ai.simulation.bank import Bank
from invest_ai.simulation.simulator import Simulator
from invest_ai.utils.string import str_to_date


def run_simulation(
    cfg: DictConfig, investor: AbstractInvestor, subset="test", prev_step=0
) -> int:
    if subset == "train":
        # limiting the training simulation to the same number of days as the test simulation
        start_date = str_to_date(cfg.experiment.time_frames[subset].start_date)
        end_date = str_to_date(cfg.experiment.time_frames[subset].end_date)
        test_len = str_to_date(
            cfg.experiment.time_frames["test"].end_date
        ) - str_to_date(cfg.experiment.time_frames["test"].start_date)
        start_date = max(end_date - test_len, start_date)
    else:
        start_date = str_to_date(cfg.experiment.time_frames[subset].start_date)
        end_date = str_to_date(cfg.experiment.time_frames[subset].end_date)
    fs = FinanceStore(
        min_date=start_date,
        max_date=end_date,
        supported_tickers=cfg.experiment.stocks,
        data_path=Path(cfg.finance_data_path),
        trading_on_weekend=cfg.experiment.bank.trading_on_weekend,
    )
    bank = hydra.utils.instantiate(cfg.experiment.bank, start_date=start_date, fs=fs)
    simulator = Simulator(
        bank=bank,
        investor=investor,
        monthly_budget=cfg.experiment.budget.monthly_budget,
    )
    simulation_days = (end_date - start_date).days
    step = prev_step
    print(f"Running {subset} simulation...")
    for i in tqdm(range(simulation_days)):
        step = prev_step + i
        if step == 35:
            print()
        status = simulator.next_day()
        stats = status.to_dict()
        stats = {f"{k}": v for k, v in stats.items()}
        # stats = {f"{k} - {subset}": v for k, v in stats.items()}
        wandb.log(stats, step=step)
    return step


def run_simulations(cfg, investor):
    last_step = run_simulation(cfg, investor, "train")
    last_step = run_simulation(cfg, investor, "val", prev_step=last_step)
    run_simulation(cfg, investor, "test", prev_step=last_step)


def preprocess_data(cfg: DictConfig) -> Tuple[Path, Path, Path]:
    train_dst, val_dst, test_dst = (
        Path(cfg.processed_data_path) / "train.pkl",
        Path(cfg.processed_data_path) / "val.pkl",
        Path(cfg.processed_data_path) / "test.pkl",
    )
    if train_dst.exists() and val_dst.exists() and test_dst.exists():
        print("Data already preprocessed. Skipping...")
        return train_dst, val_dst, test_dst
    start_date = cfg.experiment.time_frames["train"].start_date
    end_date = cfg.experiment.time_frames["test"].end_date
    fs = FinanceStore(
        min_date=start_date,
        max_date=end_date,
        supported_tickers=list(cfg.experiment.stocks),
        data_path=Path(cfg.finance_data_path),
        trading_on_weekend=cfg.experiment.bank.trading_on_weekend,
    )
    news_store = NewsStore(csv_file_path=cfg.news_data_path, backup=False)
    dp = DataPreprocessor(fs, news_store, cfg.experiment)
    train_df, val_df, test_df = dp.prepare_datasets()
    os.makedirs(cfg.processed_data_path, exist_ok=True)
    train_df.to_pickle(train_dst)
    val_df.to_pickle(val_dst)
    test_df.to_pickle(test_dst)
    return train_dst, val_dst, test_dst


@hydra.main(config_path="configs", config_name="configs")
def main(cfg: DictConfig) -> None:
    train_path, val_path, test_path = preprocess_data(cfg)
    predictor: AbstractReturnPredictor = hydra.utils.instantiate(
        cfg.experiment.predictor
    )
    try:
        logs = predictor.fit(train_path, val_path, test_path)
        if cfg.log_fine_tuning:
            wandb.init(
                project="invest-ai",
                name=cfg.experiment.experiment_name + "-fine-tuning",
                config=OmegaConf.to_container(cfg),
            )
            logs = logs.to_dict(orient="records")
            for i, log in enumerate(logs):
                log = {k: v for k, v in log.items() if not pd.isna(v)}
                wandb.log(log, step=i)
        investor = hydra.utils.instantiate(cfg.experiment.investor, predictor=predictor)
    except Exception as e:
        print(e)
        investor = hydra.utils.instantiate(cfg.experiment.investor)

    wandb.init(
        reinit=True,
        project="invest-ai",
        name=cfg.experiment.experiment_name,
        config=OmegaConf.to_container(cfg),
        tags=cfg.experiment.tags,
    )
    run_simulations(cfg, investor)


if __name__ == "__main__":
    main()
