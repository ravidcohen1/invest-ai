import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns


def plot_number_of_relevant_articles(df, dst_path: Path = None):
    df["is_relevant"] = df.keywords_count > 0
    df["year"] = pd.to_datetime(df.date).dt.to_period("Y")
    articles_count = df.groupby(df.year).keywords_count.sum()
    plt.plot(articles_count.values)
    plt.xticks(rotation=45)
    plt.xticks(range(len(articles_count.index)), articles_count.index)
    plt.title("Number of relevant articles per year")
    if dst_path:
        os.makedirs(dst_path.parent, exist_ok=True)
        plt.savefig(dst_path)
        plt.close()
    else:
        plt.show()


def plot_number_of_titles(df_path, title=None, dst: Path = None):
    df = pd.read_pickle(df_path)
    num_titles = df.title.apply(
        lambda window_titles: sum(len(daily_titles) for daily_titles in window_titles)
    )

    plt.figure(figsize=(10, 6))
    sns.histplot(num_titles, bins=10, kde=False, color=(0.2, 0.4, 0.7), alpha=0.8)
    title = title or "Distribution of Number of Titles per Window"
    plt.title(title, fontsize=20)
    plt.xlabel("Number of Titles")
    plt.ylabel("Frequency")
    if dst:
        os.makedirs(dst.parent, exist_ok=True)
        plt.savefig(dst)
        plt.close()
    else:
        plt.show()
