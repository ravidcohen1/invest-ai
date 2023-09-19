import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_return_histograms_by_target(
    df: pd.DataFrame, dst_path: Path = None, title=None
):
    """
    Plots histograms of returns grouped by the target variable.

    :param df: DataFrame containing columns 'return' and 'target'.
    :return: None
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.histplot(
        data=df,
        x="return",
        hue="target",
        bins=50,
        element="step",
        stat="percent",
        ax=ax,
    )
    title = title or "Return Distributions Grouped by Target"
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Return")
    ax.set_ylabel("Percentage of Occurrences (%)")

    plt.tight_layout()
    os.makedirs(dst_path.parent, exist_ok=True)
    if dst_path is not None:
        plt.savefig(dst_path)
        plt.close(fig)
    else:
        plt.show()


def plot_return_histograms_by_target_with_bar(df: pd.DataFrame):
    # Define bin edges and bin labels
    # bin_edges = np.linspace(df['return'].min(), df['return'].max(), 11)
    # bin_labels = [f"{left:.2f} - {right:.2f}" for left, right in zip(bin_edges[:-1], bin_edges[1:])]

    # Bin the 'return' variable
    _, bins = pd.cut(df["return"], bins=30, retbins=True)
    # df['return_bin'] = pd.cut(df['return'], bins=bin_edges, labels=bin_labels, include_lowest=True)

    # Count frequencies
    freq_data = (
        df.groupby(["return_bin", "target"]).size().reset_index(name="frequency")
    )

    # Unique bins and targets
    unique_bins = sorted(df["return_bin"].astype(str).unique())
    unique_targets = sorted(df["target"].unique())

    # Bar width and positions
    bar_width = 0.2
    index = np.arange(len(unique_bins))

    # Create the bar plot
    plt.figure(figsize=(15, 6))

    # Loop over each target to plot bars
    for i, target in enumerate(unique_targets):
        target_data = freq_data[freq_data["target"] == target]
        frequencies = [
            target_data[target_data["return_bin"] == ret_bin]["frequency"].values[0]
            if ret_bin in target_data["return_bin"].values
            else 0
            for ret_bin in unique_bins
        ]
        plt.bar(index + i * bar_width, frequencies, bar_width, label=target, alpha=0.8)

    # Add labels and title
    plt.xlabel("Return Bins")
    plt.ylabel("Frequency")
    plt.title("Frequency Count of Binned Returns by Target")
    plt.xticks(
        index + bar_width * (len(unique_targets) - 1) / 2, unique_bins, rotation=45
    )
    plt.legend()

    plt.show()


if __name__ == "__main__":
    df = pd.read_pickle(
        "/Users/user/PycharmProjects/invest-ai/results/llm_resample_titles/processed_data/train.pkl"
    )
    # plot_return_histograms_by_target(df)
    plot_return_histograms_by_target_with_bar(df)

# Sample data loading (Replace this with your actual data loading step)
# df = pd.read_pickle("../data/your_data.pkl")

# Uncomment and run the plotting function when you have loaded your DataFrame
# plot_return_histograms_by_target(df)
