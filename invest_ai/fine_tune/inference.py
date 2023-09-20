import pickle
from pathlib import Path
from typing import Dict, Optional

import openai

CACHING_DIR = Path(__file__).parent.parent.parent / "data" / "caching"


def load_cache(model: str, caching_dir: Path) -> Dict[str, str]:
    """Load cache from a pickle file."""
    caching_dir.mkdir(parents=True, exist_ok=True)
    cache_file = caching_dir / f"{model}_cache.pkl"
    try:
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


def save_cache(model: str, cache: Dict[str, str], caching_dir: Path):
    """Save cache to a pickle file."""
    caching_dir.mkdir(parents=True, exist_ok=True)
    cache_file = caching_dir / f"{model}_cache.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)


def create_completion(
    prompt: str, model: str, retries: int = 3, caching_dir: Path = CACHING_DIR
) -> Optional[str]:
    """
    Creates a completion and returns the completion.

    :param prompt: The prompt for the model.
    :param model: The model to use for the completion.
    :param retries: Number of retries for API call.
    :param retries: Path to the caching directory.
    :param caching_dir: Path to the caching directory.
    :return: Completion text or None.
    """
    # Load cache
    cache = load_cache(model, caching_dir)

    # Check cache first
    if prompt in cache:
        return cache[prompt]

    tries_left = retries
    while True:
        if tries_left == 0:
            return None
        try:
            response = openai.Completion.create(
                model=model,
                prompt=prompt,
                max_tokens=7,
                temperature=0,
                stop=["\n"],
            )
            completion = response["choices"][0]["text"].strip()

            # Update and save cache
            cache[prompt] = completion
            save_cache(model, cache, caching_dir)

            return completion

        except Exception as e:
            print(f"Error creating completion: {e}")
            tries_left -= 1
