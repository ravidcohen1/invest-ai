import datetime

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import wandb
from invest_ai.data_collection.finance_store import FinanceStore
from invest_ai.investor.abstract_investor import AbstractInvestor
from invest_ai.simulation.bank import Bank, Status
from invest_ai.simulation.simulator import Simulator


def run_simulation(simulator: Simulator, days: int = 100) -> None:
    for i in range(days):
        status = simulator.next_day()
        stats = status.to_dict()
        wandb.log(stats)


@hydra.main(config_path="../configs", config_name="configs")
def main(cfg: DictConfig) -> None:
    wandb.init(
        project="invest-ai",
        name=cfg.experiment_name,
        config=OmegaConf.to_container(cfg),
    )

    fs: FinanceStore = hydra.utils.instantiate(cfg.finance_store)
    min_date = instantiate(cfg.simulation.min_date)
    max_date = instantiate(cfg.simulation.max_date)
    bank = Bank(start_date=min_date, fs=fs, initial_amount=cfg.budget.initial_amount)
    investor: AbstractInvestor = hydra.utils.instantiate(cfg.investor)
    simulator = Simulator(
        bank=bank, investor=investor, monthly_budget=cfg.budget.monthly_budget
    )
    simulation_days = (max_date - min_date).days
    run_simulation(simulator=simulator, days=simulation_days)


if __name__ == "__main__":
    main()
