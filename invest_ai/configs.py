from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data"

FINANCE_DATA_PATH = DATA_PATH / "finance_store.pkl"
NEW_DATA_PATH = DATA_PATH / "news_store.csv"

# Sample configs.py file

from datetime import date

# Minimum and maximum supported dates for finance data
MIN_SUPPORTED_DATE = date(2007, 1, 1)
MAX_SUPPORTED_DATE = date(2023, 12, 31)

# List of supported tickers for finance data
SUPPORTED_TICKERS = [
    "AAPL",  # Apple Inc.
    "MSFT",  # Microsoft Corporation
    "AMZN",  # Amazon.com Inc.
    "GOOGL",  # Alphabet Inc.
    "META",  # Meta, Inc.
    "TCEHY",  # Tencent Holdings
    "BABA",  # Alibaba Group Holding
    "NVDA",  # NVIDIA Corporation
    "TSLA",  # Tesla, Inc.
    "PYPL",  # PayPal Holdings, Inc.
    "ADBE",  # Adobe Inc.
    "INTC",  # Intel Corporation
    "CSCO",  # Cisco Systems, Inc.
    "ORCL",  # Oracle Corporation
    "CRM",  # Salesforce.com, Inc.
    "SSNLF",  # Samsung Electronics Co., Ltd.
    "IBM",  # IBM
    "SAP",  # SAP SE
    "ASML",  # ASML Holding N.V.
    "AMD",  # Advanced Micro Devices, Inc.
    "^GSPC",  # S&P 500
]
# Default path for the CSV file to store finance data
