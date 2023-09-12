import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_return_histograms_by_target(df: pd.DataFrame):
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

    ax.set_title("Return Distributions Grouped by Target")
    ax.set_xlabel("Return")
    ax.set_ylabel("Percentage of Occurrences (%)")

    plt.tight_layout()
    plt.show()


# Sample data loading (Replace this with your actual data loading step)
# df = pd.read_pickle("../data/your_data.pkl")

# Uncomment and run the plotting function when you have loaded your DataFrame
# plot_return_histograms_by_target(df)
