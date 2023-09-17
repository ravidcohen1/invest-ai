import os
from pathlib import Path
from shutil import rmtree
from unittest.mock import patch

import pytest

from invest_ai.fine_tune.inference import create_completion, load_cache, save_cache

# Temporary directory for caching
TEMP_DIR = Path("temp_caching_dir")


@pytest.fixture
def mock_openai():
    with patch(
        "openai.Completion.create"
    ) as mock_create:  # Replace 'your_module' with the actual module name
        mock_create.return_value = {"choices": [{"text": "Mocked Response"}]}
        yield mock_create


def test_cache_hit(mock_openai):
    prompt = "Hello"
    model = "test_model"

    if TEMP_DIR.exists():
        rmtree(TEMP_DIR)

    # First Call
    response1 = create_completion(prompt, model, caching_dir=TEMP_DIR)
    assert response1 == "Mocked Response"

    # Second Call
    response2 = create_completion(prompt, model, caching_dir=TEMP_DIR)

    assert response2 == "Mocked Response"
    mock_openai.assert_called_once()


def test_cache_miss(mock_openai):
    prompt = "New Prompt"
    model = "test_model"

    # Clear existing cache
    cache = load_cache(model, TEMP_DIR)
    cache.pop(prompt, None)
    save_cache(model, cache, TEMP_DIR)

    response = create_completion(prompt, model, caching_dir=TEMP_DIR)

    assert response == "Mocked Response"
    assert mock_openai.call_count == 1


def test_retries_and_exception(mock_openai):
    prompt = "Exception"
    model = "test_model"
    mock_openai.side_effect = Exception("API Error")

    # Call the function
    response = create_completion(prompt, model, retries=3, caching_dir=TEMP_DIR)

    # Assertions
    assert response is None
    assert mock_openai.call_count == 3

    # Load cache and check if the prompt was cached
    cache = load_cache(model, TEMP_DIR)
    assert prompt not in cache


def test_cache_file_creation(mock_openai):
    model = "test_file_creation"
    cache_file = TEMP_DIR / f"{model}_cache.pkl"

    if cache_file.exists():
        os.remove(cache_file)

    create_completion("File Creation", model, caching_dir=TEMP_DIR)

    assert cache_file.exists()


def test_different_models(mock_openai):
    prompt = "Hello, world!"
    model1 = "test_model_1"
    model2 = "test_model_2"

    # Mock different responses for different models
    def side_effect(*args, **kwargs):
        if kwargs.get("model") == model1:
            return {"choices": [{"text": "Response 1"}]}
        elif kwargs.get("model") == model2:
            return {"choices": [{"text": "Response 2"}]}
        else:
            return {"choices": [{"text": "Unknown Model"}]}

    mock_openai.side_effect = side_effect

    # Call the function with different models
    response1 = create_completion(prompt, model1, caching_dir=TEMP_DIR)
    response2 = create_completion(prompt, model2, caching_dir=TEMP_DIR)

    # Assertions
    assert response1 == "Response 1"
    assert response2 == "Response 2"

    # Load cache and check if the prompt was cached separately for each model
    cache1 = load_cache(model1, TEMP_DIR)
    cache2 = load_cache(model2, TEMP_DIR)

    assert cache1[prompt] == "Response 1"
    assert cache2[prompt] == "Response 2"
