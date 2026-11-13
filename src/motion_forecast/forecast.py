"""Forecast the Motionen the Federal Council has not answered yet, and score the earlier
forecasts it has answered since.

    PARL_API_KEY=... python -m motion_forecast.forecast

Run it every few weeks: the Federal Council answers after about two months, so a run
within a few weeks of each session forecasts nearly every Motion before its answer.
Each run downloads the latest data, forecasts each open Motion once with the champion
model, scores the forecasts that have been answered, logs both to MLflow and commits the
new forecasts to git.

forecasts.csv is append-only: a forecast is never changed, and its commit shows it was
made before the answer.
"""

import subprocess
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import brier_score_loss, log_loss

from motion_forecast.data import load, update
from motion_forecast.models import MODEL_NAME, RAW

FORECASTS_PATH = Path("forecasts.csv")
COLUMNS = ["id", "number", "title", "forecasted", "model_version", "p_accept"]
EXPERIMENT = "live-forecast"
# Older Motionen without an answer will not get one any more (withdrawn, written off).
OPEN_FOR = pd.Timedelta(days=180)


def open_motions(df: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """Submitted recently and still waiting for the Federal Council's answer."""
    waiting = df["proposal_date"].isna() & (df["status"] != "Erledigt")
    return df[waiting & (df["submitted"] >= today - OPEN_FOR)]


def forecast_new(
    df: pd.DataFrame, done: pd.DataFrame, model, version: str, today: pd.Timestamp
) -> pd.DataFrame:
    """Forecasts for the open Motionen that have none yet."""
    todo = open_motions(df, today)
    todo = todo[~todo["id"].isin(done["id"])]
    if todo.empty:
        return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(
        {
            "id": todo["id"],
            "number": todo["number"],
            "title": todo["title"],
            "forecasted": str(today.date()),
            "model_version": version,
            "p_accept": model.predict(todo[RAW]).round(4),
        }
    )


def score_answered(df: pd.DataFrame, forecasts: pd.DataFrame) -> dict[str, float]:
    """How the forecasts did, now that the Federal Council has answered them."""
    answered = forecasts.merge(df[["id", "label"]], on="id").dropna(subset="label")
    if answered.empty:
        return {"answered": 0}
    y, p = answered["label"], answered["p_accept"]
    return {
        "answered": len(answered),
        "live_log_loss": log_loss(y, p, labels=[0, 1]),
        "live_brier": brier_score_loss(y, p),
        # Calibration in one number: are as many accepted as forecast?
        "predicted_rate": p.mean(),
        "actual_rate": y.mean(),
    }


def publish(n: int) -> None:
    """Commit only the forecasts file, so the push timestamps them publicly."""
    path = str(FORECASTS_PATH)
    subprocess.run(["git", "add", path], check=True)
    subprocess.run(["git", "commit", "-m", f"Forecast {n} Motionen", "--", path], check=True)
    subprocess.run(["git", "push"], check=True)


def main() -> None:
    update()
    df = load()
    today = pd.Timestamp.today().normalize()
    done = pd.read_csv(FORECASTS_PATH) if FORECASTS_PATH.exists() else pd.DataFrame(columns=COLUMNS)

    # Resolve the alias once, so the version recorded is the one that made the forecasts.
    champion = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{champion.version}")
    new = forecast_new(df, done, model, champion.version, today)
    forecasts = pd.concat([done, new], ignore_index=True)
    metrics = score_answered(df, forecasts)
    # The champion's score on the test set: live scores far above it mean the model has
    # gone stale, and it is time to retrain.
    test_log_loss = MlflowClient().get_run(champion.run_id).data.metrics["log_loss"]

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=str(today.date())):
        mlflow.log_param("model_version", champion.version)
        mlflow.log_metrics({"new_forecasts": len(new), "test_log_loss": test_log_loss, **metrics})

    print(f"{len(new)} new forecasts by version {champion.version}, {len(forecasts)} in total")
    print(f"answered so far: {metrics['answered']}")
    if metrics["answered"]:
        print(f"live log loss {metrics['live_log_loss']:.3f} (test set: {test_log_loss:.3f})")
        print(f"accepted: {metrics['actual_rate']:.0%}, forecast {metrics['predicted_rate']:.0%}")
    if len(new):
        forecasts.to_csv(FORECASTS_PATH, index=False)
        publish(len(new))


if __name__ == "__main__":
    main()
