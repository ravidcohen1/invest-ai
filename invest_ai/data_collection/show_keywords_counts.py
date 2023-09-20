import matplotlib.pyplot as plt

from invest_ai.data_collection.news_store import NewsStore
from invest_ai.data_collection.tickers_keywords import TICKER_KEYWORDS
from tqdm import tqdm
import pandas as pd


def plot_number_of_relevant_articles():
    articles_count_sums = {}
    news_store = NewsStore(backup=False)
    for k, v in tqdm(TICKER_KEYWORDS.items()):
        news_store.keywords = v
        df = news_store.get_news_for_dates(
            "2007-01-01", "2023-09-01", fetch_missing_dates=False
        )
        df["is_relevant"] = df.keywords_count > 0
        df["week"] = pd.to_datetime(df.date).dt.to_period("Y")
        articles_count = df.groupby(df.week).keywords_count.sum()
        plt.plot(articles_count.values, label=k)
        plt.xticks(rotation=45)
        # set x tics labels
        plt.xticks(range(len(articles_count.index)), articles_count.index)

        articles_count_sums[k] = (df.keywords_count > 0).sum()
    plt.legend()
    plt.show()

    articles_count_sums = dict(
        sorted(articles_count_sums.items(), key=lambda item: item[1])
    )
    plt.bar(articles_count_sums.keys(), articles_count_sums.values())
    plt.xticks(rotation=45)
    plt.show()


if __name__ == "__main__":
    plot_number_of_relevant_articles()
