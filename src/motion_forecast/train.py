"""Run the model ladder and track every run with MLflow.

    python -m motion_forecast.train
    mlflow ui                          # then open http://localhost:5000

Each stage is one MLflow run. Inside it, one nested run per hyperparameter setting is
trained on 2007-2018 and scored on 2019-2022 (validation). The best setting is then
retrained on 2007-2022 and scored once on 2023 onwards (test), in the stage's own run.
MLflow stores everything in mlflow.db (SQLite) and the artifacts in mlruns/.
"""

import matplotlib

matplotlib.use("Agg")  # draw to files, never open a window

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from motion_forecast.data import load
from motion_forecast.models import LADDER, inputs

EXPERIMENT = "model-ladder"
FIRST = pd.Timestamp("2007-01-01")  # the acceptance rate fell from ~60% in the 1990s
VALIDATION = pd.Timestamp("2019-01-01")
TEST = pd.Timestamp("2023-01-01")


def periods(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split by time, the way the model would have been used.

    A model trained on 1 January only knows the answers given before that day. Motionen
    submitted in December but answered in January belong to no training set.
    """
    df = df.dropna(subset="label")
    df = df[df["submitted"] >= FIRST]
    submitted, answered = df["submitted"], df["proposal_date"]
    return {
        "train": df[answered < VALIDATION],
        "validation": df[(submitted >= VALIDATION) & (submitted < TEST)],
        "train_validation": df[answered < TEST],
        "test": df[submitted >= TEST],
    }


def score(y: pd.Series, p: np.ndarray) -> dict[str, float]:
    # Not accuracy: "always reject" is right 80% of the time and useless.
    return {
        "log_loss": log_loss(y, p, labels=[0, 1]),  # main metric: punishes confident mistakes
        "brier": brier_score_loss(y, p),  # mean squared error of the probability
        "pr_auc": average_precision_score(y, p),  # how well it finds the rare accepts
        "roc_auc": roc_auc_score(y, p),  # how well it ranks accepts above rejects
    }


def curves(y: pd.Series, p: np.ndarray) -> plt.Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.5))
    # Calibration: of the Motionen given 30%, are 30% accepted?
    CalibrationDisplay.from_predictions(y, p, n_bins=10, strategy="quantile", ax=left)
    PrecisionRecallDisplay.from_predictions(y, p, ax=right, plot_chance_level=True)
    left.set_title("Calibration")
    right.set_title("Precision and recall of 'accept'")
    fig.tight_layout()
    return fig


def by_party(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Share accepted per party, as predicted and as it happened."""
    table = df.assign(predicted=p).groupby("party")
    return pd.DataFrame(
        {
            "motions": table.size(),
            "actual": table["label"].mean(),
            "predicted": table["predicted"].mean(),
        }
    ).reset_index()


def fit_and_score(stage: str, params: dict, fit: pd.DataFrame, evaluate: pd.DataFrame) -> dict:
    """Train on one period, score on a later one, and log both data sets to the active run."""
    make_model, _ = LADDER[stage]
    model = make_model(**params).fit(inputs(fit), fit["label"])
    p = model.predict_proba(inputs(evaluate))[:, 1]

    # Which rows the run saw: MLflow stores a digest, so a changed download shows up.
    for name, part in [("fit", fit), ("evaluate", evaluate)]:
        dataset = mlflow.data.from_pandas(part[["number", "label"]], name=name, targets="label")
        mlflow.log_input(dataset, context=name)
    mlflow.log_params(params)
    metrics = score(evaluate["label"], p)
    mlflow.log_metrics(metrics)
    return {"metrics": metrics, "p": p}


def run_stage(stage: str, data: dict[str, pd.DataFrame]) -> dict[str, float]:
    _, grid = LADDER[stage]
    with mlflow.start_run(run_name=stage, tags={"stage": stage}):
        # Choose the hyperparameters on validation, one nested run each.
        results = []
        for params in grid:
            with mlflow.start_run(run_name=f"{stage} {params}", nested=True):
                result = fit_and_score(stage, params, data["train"], data["validation"])
                results.append((result["metrics"]["log_loss"], params))
        best = min(results, key=lambda r: r[0])[1]

        # Retrain on everything known by the test period, score once on the test.
        result = fit_and_score(stage, best, data["train_validation"], data["test"])
        fig = curves(data["test"]["label"], result["p"])
        mlflow.log_figure(fig, "curves.png")
        plt.close(fig)
        mlflow.log_table(by_party(data["test"], result["p"]), "by_party.json")
        return result["metrics"]


def main() -> None:
    data = periods(load())
    for name, part in data.items():
        print(f"{name}: {len(part)} Motionen, {part['label'].mean():.1%} accepted")

    # No mlflow.sklearn.autolog(): after every fit it scores the training data and draws
    # ROC, PR and confusion plots, which made each run ~7x slower. Training scores say
    # little; what matters is logged by hand above, on held-out data only.
    mlflow.set_experiment(EXPERIMENT)
    results = {stage: run_stage(stage, data) for stage in LADDER}
    table = pd.DataFrame(results).T
    print("\nTest set (2023 onwards):")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
