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
        for s in cfg.experiment.stocks:
            if s not in stats["portfolio"]:
                stats["portfolio"][s] = 0
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
    from invest_ai.data_collection.tickers_keywords import TICKER_KEYWORDS

    keywords = []
    for s in cfg.experiment.stocks:
        keywords += TICKER_KEYWORDS[s]
    news_store = NewsStore(
        csv_file_path=cfg.news_data_path, backup=False, keywords=keywords
    )
    df = news_store.get_news_for_dates(start_date, end_date, fetch_missing_dates=False)
    from invest_ai.plots.articles import plot_number_of_relevant_articles

    plot_number_of_relevant_articles(
        df,
        dst_path=Path(cfg.experiment_path)
        / "plots"
        / "number_of_relevant_articles.png",
    )
    dp = DataPreprocessor(fs, news_store, cfg.experiment)
    train_df, val_df, test_df = dp.prepare_datasets()
    os.makedirs(cfg.processed_data_path, exist_ok=True)
    train_df.to_pickle(train_dst)
    val_df.to_pickle(val_dst)
    test_df.to_pickle(test_dst)
    return train_dst, val_dst, test_dst


from invest_ai.plots.histograms import plot_return_histograms_by_target
from imageio.v3 import imread
import numpy as np
from skimage.transform import resize


def plot_histograms(cfg, train_path, val_path, test_path):
    histograms = []
    for df_path, subset in zip(
        [train_path, val_path, test_path], ["train", "val", "test"]
    ):
        df = pd.read_pickle(df_path)
        dst_path = (
            Path(cfg.experiment_path) / "plots" / f"{subset}_returns_histogram.png"
        )
        plot_return_histograms_by_target(df, dst_path=dst_path, title=subset)
        img = imread(dst_path)
        histograms.append(img)

    histograms = np.concatenate(histograms, axis=1)
    w = 1200
    h = int(histograms.shape[0] * w / histograms.shape[1])
    histograms = resize(histograms, (h, w), anti_aliasing=True)
    wandb.log({"returns_histogram": wandb.Image(histograms)})

    articles_path = (
        Path(cfg.experiment_path) / "plots" / "number_of_relevant_articles.png"
    )
    if articles_path.exists():
        img = imread(articles_path)
        wandb.log({"number_of_relevant_articles": wandb.Image(img)})

    from invest_ai.plots.articles import plot_number_of_titles

    titles_path_train = (
        Path(cfg.experiment_path) / "plots" / "number_of_titles_train.png"
    )
    plot_number_of_titles(
        train_path, title="Number of titles per week - train", dst=titles_path_train
    )
    img = imread(titles_path_train)
    wandb.log({"number_of_titles_train": wandb.Image(img)})

    titles_path_test = Path(cfg.experiment_path) / "plots" / "number_of_titles_test.png"
    plot_number_of_titles(
        test_path, title="Number of titles per week - test", dst=titles_path_test
    )
    img = imread(titles_path_test)
    wandb.log({"number_of_titles_test": wandb.Image(img)})


@hydra.main(config_path="configs", config_name="configs")
def main(cfg: DictConfig) -> None:
    train_path, val_path, test_path = preprocess_data(cfg)
    if "predictor" in cfg.experiment:
        predictor: AbstractReturnPredictor = hydra.utils.instantiate(
            cfg.experiment.predictor
        )
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
            plot_histograms(cfg, train_path, val_path, test_path)
        investor = hydra.utils.instantiate(cfg.experiment.investor, predictor=predictor)
    else:
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
