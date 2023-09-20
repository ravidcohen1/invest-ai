import json
import time
from io import StringIO
from pathlib import Path
from typing import List, Optional, Tuple

import openai
import pandas as pd


def upload_file_until_processed(file_path: Path, caching_dir: Path) -> str:
    """Uploads a file to the OpenAI API and returns the file id."""
    dst_path = caching_dir / f"{file_path.stem} - upload-response.json"
    if dst_path.exists():
        print(f"File {file_path} already uploaded. Skipping File.create...")
        with open(dst_path, "r") as f:
            response = json.load(f)
    else:
        print("Uploading file...")
        response = openai.File.create(file=open(file_path, "rb"), purpose="fine-tune")
        file_id = response.id
        print("Waiting for file process to finish...")
        while response.status != "processed":
            if response.status == "failed":
                raise ValueError(f"File upload failed:\n{response}")
            time.sleep(2)
            response = openai.File.retrieve(file_id)

        with open(dst_path, "w") as f:
            json.dump(response, f)
    return response["id"]


def create_fine_tuning_job(
    training_file_id: str,
    validation_file_id: str,
    caching_dir: Path,
    model: str,
    epochs: Optional[int] = None,
) -> dict:
    """Creates a fine-tuning job and returns the job id."""
    job_path = caching_dir / f"fine-tune-response.json"
    if job_path.exists():
        print(f"Fine-tuning job already created. Skipping FineTuningJob.create...")
        with open(job_path, "r") as f:
            job = json.load(f)
            print(job)
    else:
        print("Creating fine-tuning job...")
        hyperparameters = {"n_epochs": epochs} if epochs else {}
        job = openai.FineTuningJob.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model=model,
            hyperparameters=hyperparameters,
        )
        with open(job_path, "w") as f:
            json.dump(job, f)
        print(f"Fine-tuning job created:\n{job}\n\nAnd saved to {job_path}")

    if job["status"] == "succeeded":
        return job
    print("Waiting for fine-tuning job to finish...")
    prev_status = ""
    while job["status"] != "succeeded":
        if job["status"] != prev_status:
            print(f"Current status: {job['status']}")
            prev_status = job["status"]
        if job["status"] == "failed":
            raise ValueError(f"Fine-tuning job failed:\n{job}")
        time.sleep(5)
        job = openai.FineTuningJob.retrieve(job["id"])
    print(f"Fine-tuning job finished!\n{job}")
    with open(job_path, "w") as f:
        json.dump(job, f)
    return job


def convert_to_jsonl(df_path: Path) -> Path:
    new_path = df_path.with_suffix(".jsonl")
    if not new_path.exists():
        json_data = pd.read_csv(df_path).to_json(orient="records", lines=True)
        with open(new_path, "w") as f:
            f.write(json_data)
    return new_path


def get_logs(file_id: str, caching_dir: Path) -> pd.DataFrame:
    dst_path = caching_dir / "logs.csv"
    if dst_path.exists():
        df = pd.read_csv(dst_path)
    else:
        byte_data = openai.File.download(file_id)
        decoded_data = byte_data.decode("utf-8")
        df = pd.read_csv(StringIO(decoded_data))
        df.to_csv(dst_path)
    return df


def validate_files(train_path: Path, val_path: Path, labels: List[str]) -> None:
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    assert "prompt" in train_df.columns
    assert "completion" in train_df.columns
    assert "index" in train_df.columns
    assert train_df.shape[0] > 0, "Train data is empty"
    assert val_df.shape[0] > 0, "Val data is empty"
    assert set(train_df["completion"].apply(lambda x: x.strip()).unique()) == set(
        labels
    ), "Labels don't match"
    assert set(val_df["completion"].apply(lambda x: x.strip()).unique()) == set(
        labels
    ), "Labels don't match"


def fine_tune_and_wait(
    train_path: Path,
    val_path: Path,
    caching_dir: Path,
    model: str,
    epochs: Optional[int],
    labels: List[str],
) -> Tuple[str, pd.DataFrame]:
    assert train_path.suffix == ".csv" and val_path.suffix == ".csv"
    validate_files(train_path, val_path, labels)
    train_path = convert_to_jsonl(train_path)
    val_path = convert_to_jsonl(val_path)
    train_file_id = upload_file_until_processed(train_path, caching_dir)
    val_file_id = upload_file_until_processed(val_path, caching_dir)
    job = create_fine_tuning_job(
        train_file_id, val_file_id, caching_dir, model=model, epochs=epochs
    )
    model_id = job["fine_tuned_model"]
    result_files = job["result_files"]
    assert len(result_files) == 1, "Expected only one result file."
    logs = get_logs(result_files[0], caching_dir)

    return model_id, logs
