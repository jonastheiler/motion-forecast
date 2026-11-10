# motion-forecast

Predicts whether the Swiss Federal Council will propose to accept or reject a new Motion
or Postulat, from what is known when it is submitted: its text, its author's party, the
council and the department. A project to learn the MLflow model lifecycle.

Unofficial. Data: [Swiss Parliament open data](https://www.parlament.ch/de/%C3%BCber-das-parlament/fakten-und-zahlen/open-data-web-services) (ws.parlament.ch).

## Run

```
uv sync
uv run python -m motion_forecast.data   # download ~20k Motionen and Postulate to data/
uv run pytest
```

## Status

- [x] M0 setup, M1 data
- [ ] M2 baselines and experiment tracking with MLflow
- [ ] M3 embeddings and a fine-tuned transformer
- [ ] M4 model registry and pyfunc
- [ ] M5 serving
- [ ] M6 weekly live forecast and monitoring
