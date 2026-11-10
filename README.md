# motion-forecast

Predicts whether the Swiss Federal Council will propose to accept or reject a new Motion
or Postulat, from what is known when it is submitted: its text, its author's party, the
council and the department. A project to learn the MLflow model lifecycle.

Unofficial. Data: [Swiss Parliament open data](https://www.parlament.ch/de/%C3%BCber-das-parlament/fakten-und-zahlen/open-data-web-services) (ws.parlament.ch).

## Run

```
uv sync
uv run python -m motion_forecast.data    # download ~20k Motionen and Postulate (~2 min)
uv run python -m motion_forecast.train   # model ladder, tracked in MLflow (~3 min)
uv run mlflow ui                         # compare the runs at http://localhost:5000
uv run pytest
```

## Results so far

Trained on Motionen and Postulate answered 2007-2022, tested on those submitted since
2023 (3,116, of which 20% accepted). Lower log loss is better.

| Model | Log loss | Brier | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| Majority (always 20%) | 0.506 | 0.162 | 0.20 | 0.50 |
| Metadata (party, council, type, department) | 0.431 | 0.136 | 0.46 | 0.76 |
| Text (TF-IDF) | 0.444 | 0.142 | 0.43 | 0.74 |
| Text + metadata | **0.414** | **0.131** | **0.50** | **0.79** |

Who asks says more than what they ask: metadata alone beats the text alone. Committee
Motionen are accepted half the time, SVP Motionen 9%. The words still add information
on top.

## Key ideas

- **Split by time, train on what was known.** A model trained on 1 January only knows
  the answers given before that day (`train.periods`).
- **Accuracy misleads at 80/20.** "Always reject" is 80% accurate. Log loss and Brier
  score the probabilities, PR-AUC the rare accepts.
- **MLflow tracking.** One run per stage, with a nested run per hyperparameter setting.
  Each run logs its parameters, metrics, a digest of its data, a calibration and PR plot,
  and a per-party table.
- **Autologging has a cost.** `mlflow.sklearn.autolog()` made each run ~7x slower by
  scoring and plotting the training data, so the code logs only held-out scores.

## Status

- [x] M0 setup, M1 data
- [x] M2 baselines and experiment tracking with MLflow
- [ ] M3 embeddings and a fine-tuned transformer
- [ ] M4 model registry and pyfunc
- [ ] M5 serving
- [ ] M6 weekly live forecast and monitoring
