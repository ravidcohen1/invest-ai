from pathlib import Path
from typing import Tuple

import pandas as pd

from invest_ai.utils.string import date_to_str


# Helper function to format float numbers for better readability
def format_float_list(float_list):
    return ", ".join([f"{x:.4f}" if x is not None else "nan" for x in float_list])


# Prepare 1-2 examples again, this time with formatted floats
def prepare_sample_for_prompt_with_format(sample: pd.Series) -> Tuple[str, str, str]:
    prompt_data = {
        "ticker": sample["ticker"],
        "weekday": ", ".join(sample["weekday"]),
        "open": format_float_list(sample["open"]),
        "close": format_float_list(sample["close"]),
        "adj_close": format_float_list(sample["adj_close"]),
        "high": format_float_list(sample["high"]),
        "low": format_float_list(sample["low"]),
        "volume": format_float_list(sample["volume"]),
        "title": "\n".join(
            [str(t) if isinstance(t, list) else "" for t in sample["title"]]
        ),  # Shortening titles for brevity
    }

    prompt = f"""
            Ticker: {prompt_data['ticker']}
            Weekdays: {prompt_data['weekday']}
            Stock Data:
            - Open Prices: {prompt_data['open']}
            - Close Prices: {prompt_data['close']}
            - Adjusted Close Prices: {prompt_data['adj_close']}
            - High Prices: {prompt_data['high']}
            - Low Prices: {prompt_data['low']}
            - Volume: {prompt_data['volume']}
            News Titles:
            {prompt_data['title']}
            Predict the expected stock return for the next day:
    """
    index = get_index_from_sample(sample)
    completion = f"{sample['target']}\n"
    return index, prompt, completion


def get_index_from_sample(sample: pd.Series) -> str:
    stock_key = "ticker" if "ticker" in sample else "stock"
    if isinstance(sample["date"], list):
        date = sample["date"][-1]
    else:
        date = sample["date"]
    return f"{sample[stock_key]} - {date_to_str(date)}"


# Function to generate a pd.Series for the prompt and another for the completion given a DataFrame
def generate_prompt_and_completion_series(
    data_path: Path, dst_path: Path
) -> pd.DataFrame:
    """
    Generate two pd.Series: one for the prompts and another for the completions, based on the DataFrame.

    Parameters:
        data_path : The path to the pickled DF
        dst_path : path to the destination csv file

    Returns:
        pd.Series: Series containing the formatted prompts.
        pd.Series: Series containing the corresponding completions (target labels).
    """
    assert data_path.suffix == ".pkl", "Data path must be a pickled DataFrame"
    assert dst_path.suffix == ".csv", "Destination path must be a csv file"
    df = pd.read_pickle(data_path)
    prompts = []
    completions = []
    indexes = []

    for _, row in df.iterrows():
        index, prompt, completion = prepare_sample_for_prompt_with_format(row)
        prompts.append(prompt)
        completions.append(completion)
        indexes.append(index)

    ft_df = pd.DataFrame(
        {
            "prompt": pd.Series(prompts),
            "completion": pd.Series(completions),
            "index": pd.Series(indexes),
        }
    )
    ft_df.to_csv(dst_path, index=False)
    print(
        f"Saved fine-tuning data to {dst_path}, with {len(ft_df)} samples, null rows: {ft_df.isnull().sum()}"
    )
    return ft_df
