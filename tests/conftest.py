from datetime import date
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from invest_ai.data_collection.finance_store import FinanceStore
from invest_ai.data_collection.news_store import NewsStore
from invest_ai.data_collection.preprocess import DataPreprocessor
from invest_ai.simulation.bank import Bank

FINANCE_DATA_PATH = Path(__file__).parent.absolute() / "data/test_finance_store.pkl"
NEWS_DATA_PATH = Path(__file__).parent.absolute() / "data/test_news_store.csv"
CONFIGS_FILE = Path(__file__).parent.absolute() / "test_data_configs.yaml"


# Setup fixtures for reusable components
@pytest.fixture
def news_store():
    ns = NewsStore(NEWS_DATA_PATH, backup=False, keywords=["is"])
    if not NEWS_DATA_PATH.exists():
        ns.get_news_for_dates("2023-08-01", "2023-08-30", fetch_missing_dates=True)
    return ns


@pytest.fixture(scope="module")
def finance_store():
    fs = FinanceStore(
        min_date=date(2023, 8, 1),
        max_date=date(2023, 8, 29),
        supported_tickers=["AAPL", "GOOGL"],
        data_path=FINANCE_DATA_PATH,
        trading_on_weekend=False,
    )
    return fs


@pytest.fixture(scope="module")
def finance_store_weekends():
    fs = FinanceStore(
        min_date=date(2023, 8, 1),
        max_date=date(2023, 8, 29),
        supported_tickers=["AAPL", "GOOGL"],
        data_path=FINANCE_DATA_PATH,
        trading_on_weekend=True,
    )
    return fs


@pytest.fixture(scope="module")
def bank(finance_store: FinanceStore) -> Bank:
    return Bank(start_date=date(2023, 8, 15), fs=finance_store, initial_amount=10000)


@pytest.fixture
def setup_data_preprocessor(news_store, finance_store):
    config = OmegaConf.load(CONFIGS_FILE)
    return DataPreprocessor(finance_store, news_store, config)
