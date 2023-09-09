import pandas as pd
import yfinance as yf


def get_stock_prices(tickers, start_date, end_date) -> pd.DataFrame:
    """

    :param tickers: str, list
    :param start_date: YYYY-MM-DD
    :param end_date: YYYY-MM-DD
    :return:
    """
    data = yf.download(tickers, start=start_date, end=end_date, ignore_tz=True)
    return data


# stock_symbols = ["GOOGL", "MSFT", "AAPL", "AMZN", "TSLA", "FB", "NFLX", "BABA", "JNJ", "JPM", "V", "PG", "BAC", "INTC", "KO", "MCD", "NKE", "T", "PFE", "ORCL"]
# List of ticker symbols for the 20 largest tech companies based on market capitalization as of mid-2021
stock_symbols = [
    "AAPL",  # Apple Inc.
    "MSFT",  # Microsoft Corporation
    "AMZN",  # Amazon.com Inc.
    "GOOGL",  # Alphabet Inc.
    "FB",  # Facebook, Inc.
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
]


# stock_symbols = ["^GSPC"] # this is s&p 500
df = get_stock_prices(stock_symbols, "2007-01-01", "2023-09-01")
print(df)
df.to_csv("snp.csv")
from matplotlib import pyplot as plt

if len(stock_symbols) == 1:
    plt.plot(df["Close"])
    plt.title(stock_symbols[0])
else:
    for s in stock_symbols:
        plt.plot(df["Close"][s], label=s)
    plt.legend()
plt.show()


import pandas_datareader as pdr


def get_all_tickers() -> list:
    """
    Get all tickers using pandas_datareader.

    :return: List of all tickers
    :rtype: list
    """
    df = pdr.get_nasdaq_symbols()
    return df.index.tolist()


# Get all tickers
all_tickers = get_all_tickers()
