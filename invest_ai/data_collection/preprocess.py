def categorize_returns_equal_frequency(returns, n_tiers):
    """
    Categorizes returns into n discrete tiers using equal frequency binning.

    :param returns: Pandas Series of returns.
    :param n_tiers: Number of tiers to split the returns into.

    :return: Categorized returns as a Pandas Series.
    """
    # Compute quantile bin edges
    bin_edges = np.linspace(0, 1, n_tiers + 1)
    quantile_bins = returns.quantile(bin_edges)

    # Use 'cut' function to categorize returns
    categories = pd.cut(
        returns, bins=quantile_bins, labels=range(1, n_tiers + 1), include_lowest=True
    )

    return categories
