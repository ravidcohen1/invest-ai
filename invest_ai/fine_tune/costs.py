from functools import partial

import pandas as pd
import tiktoken

TRAINING_COSTS_PER_1000 = {"davinci-002": 0.006, "babbage-002": 0.0004}
INFERENCE_COSTS_PER_1000 = {"davinci-002": 0.012, "babbage-002": 0.0016}


def compute_training_costs(df: pd.DataFrame, model, epochs):
    assert "prompt" in df.columns
    assert "completion" in df.columns
    epochs = epochs or 4
    prompt_tokens = df["prompt"].apply(partial(compute_tokens, model=model)).sum()
    completion_tokens = (
        df["completion"].apply(partial(compute_tokens, model=model)).sum()
    )
    return (
        (prompt_tokens + completion_tokens)
        * TRAINING_COSTS_PER_1000[model]
        * epochs
        / 1000
    )


def compute_inference_costs(df: pd.DataFrame, model):
    assert "prompt" in df.columns
    assert "completion" in df.columns
    prompt_tokens = df["prompt"].apply(partial(compute_tokens, model=model)).sum()
    completion_tokens = (
        df["completion"].apply(partial(compute_tokens, model=model)).sum()
    )
    return (prompt_tokens + completion_tokens) * INFERENCE_COSTS_PER_1000[model] / 1000


def compute_tokens(text, model):
    return len(tiktoken.encoding_for_model(model).encode(text))
