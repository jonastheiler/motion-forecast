"""Fine-tune the pretrained language model on the labels: the ladder's last stage.

    python -m motion_forecast.finetune train   # on a GPU (free Kaggle: see the README)
    python -m motion_forecast.finetune log     # back here: log the result to MLflow

The embedding stages feed the frozen vectors of multilingual-e5-small to a logistic
regression. Here the same model is trained end to end, with a small classification
layer on top, so the gap between the two stages is what fine-tuning is worth.

`train` follows the ladder's protocol: train on 2007-2018 and choose the number of epochs
on validation, then retrain on 2007-2022 for that many epochs and predict the test set.
It writes the predictions to data/finetuned.parquet, and `log` scores them like every
other stage. Not deployable: serving it would need the language model.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from motion_forecast.data import load
from motion_forecast.embed import MODEL
from motion_forecast.models import WORDS, inputs
from motion_forecast.train import EXPERIMENT, by_party, curves, periods, score

PREDICTIONS_PATH = Path("data/finetuned.parquet")
MAX_EPOCHS = 3
BATCH = 16
# Small steps: large ones would overwrite what the model learned in pretraining.
LEARNING_RATE = 3e-5
MAX_TOKENS = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HALF = DEVICE == "cuda"  # 16-bit floats on a GPU: about twice as fast, same results


def batches(df: pd.DataFrame, tokenizer, shuffle: bool) -> DataLoader:
    """The Motionen's words as token ids, BATCH at a time, with their labels."""
    rows = list(zip(inputs(df)[WORDS], df["label"].astype(int), strict=True))

    def collate(batch: list) -> dict:
        texts, labels = zip(*batch, strict=True)
        tokens = tokenizer(
            list(texts), truncation=True, max_length=MAX_TOKENS, padding=True, return_tensors="pt"
        )
        return {**tokens, "labels": torch.tensor(labels)}

    return DataLoader(rows, batch_size=BATCH, shuffle=shuffle, collate_fn=collate)


def predict(model, tokenizer, df: pd.DataFrame) -> np.ndarray:
    """Probability of acceptance for each Motion."""
    model.eval()
    probabilities = []
    with torch.no_grad():
        for batch in batches(df, tokenizer, shuffle=False):
            batch = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=HALF):
                logits = model(**batch).logits
            probabilities.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities)


def fit(train: pd.DataFrame, epochs: int, evaluate: pd.DataFrame | None = None) -> tuple:
    """Fine-tune a fresh copy of the pretrained model. After each epoch, predict `evaluate`."""
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(DEVICE)
    data = batches(train, tokenizer, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    steps = epochs * len(data)
    # Warm up over the first 10% of the steps, then let the step size fall to zero.
    schedule = get_linear_schedule_with_warmup(optimizer, steps // 10, steps)
    scaler = torch.amp.GradScaler(enabled=HALF)  # keeps 16-bit gradients from underflowing

    per_epoch = []
    for epoch in range(epochs):
        model.train()
        for batch in data:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=HALF):
                loss = model(**batch).loss  # cross-entropy against the labels
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            schedule.step()
        if evaluate is not None:
            p = predict(model, tokenizer, evaluate)
            per_epoch.append(p)
            loss = score(evaluate["label"], p)["log_loss"]
            print(f"epoch {epoch + 1}: validation log loss {loss:.3f}")
    return model, tokenizer, per_epoch


def train() -> None:
    data = periods(load())
    print(f"fine-tuning {MODEL} on {DEVICE}")
    # Choose the number of epochs on validation: more epochs fit the training years
    # closer, until the model starts to memorise them.
    _, _, per_epoch = fit(data["train"], MAX_EPOCHS, evaluate=data["validation"])
    losses = [score(data["validation"]["label"], p)["log_loss"] for p in per_epoch]
    epochs = int(np.argmin(losses)) + 1

    model, tokenizer, _ = fit(data["train_validation"], epochs)
    predictions = pd.concat(
        [
            pd.DataFrame(
                {"id": data["validation"]["id"], "split": "validation", "p": per_epoch[epochs - 1]}
            ),
            pd.DataFrame(
                {
                    "id": data["test"]["id"],
                    "split": "test",
                    "p": predict(model, tokenizer, data["test"]),
                }
            ),
        ]
    ).assign(epochs=epochs)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(PREDICTIONS_PATH, index=False)
    print(f"{epochs} epochs; predictions written to {PREDICTIONS_PATH}")


def log() -> None:
    data = periods(load())
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    parts = {}
    for split in ["validation", "test"]:
        part = data[split].merge(predictions[predictions["split"] == split], on="id")
        if len(part) != len(data[split]):
            raise SystemExit(f"{PREDICTIONS_PATH} was made from a different download of the data")
        parts[split] = part

    test = parts["test"]
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="finetuned", tags={"stage": "finetuned"}):
        mlflow.log_params(
            {
                "model": MODEL,
                "epochs": int(predictions["epochs"].iloc[0]),
                "learning_rate": LEARNING_RATE,
                "max_tokens": MAX_TOKENS,
            }
        )
        validation = parts["validation"]
        mlflow.log_metric(
            "validation_log_loss", score(validation["label"], validation["p"])["log_loss"]
        )
        metrics = score(test["label"], test["p"])
        mlflow.log_metrics(metrics)
        fig = curves(test["label"], test["p"])
        mlflow.log_figure(fig, "curves.png")
        plt.close(fig)
        mlflow.log_table(by_party(test, test["p"].to_numpy()), "by_party.json")
    print("Test set (2023 onwards):", {k: round(v, 3) for k, v in metrics.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["train", "log"])
    {"train": train, "log": log}[parser.parse_args().step]()
