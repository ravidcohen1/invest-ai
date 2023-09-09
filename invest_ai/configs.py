from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data"

FINANCE_DATA_PATH = DATA_PATH / "finance_store.pkl"
NEW_DATA_PATH = DATA_PATH / "news_store.csv"

# Sample configs.py file

from datetime import date

# Minimum and maximum supported dates for finance data
MIN_SUPPORTED_DATE = date(2020, 1, 1)
MAX_SUPPORTED_DATE = date(2023, 12, 31)

# List of supported tickers for finance data
SUPPORTED_TICKERS = ["AAPL", "GOOGL", "MSFT"]

# Default path for the CSV file to store finance data
