"""The HTTP contract of `mlflow models serve`, checked in-process: the server is a
FastAPI app, so a test client can call it without starting a web server."""

import json

import mlflow
import pytest
from fastapi.testclient import TestClient
from mlflow.pyfunc import scoring_server
from test_train import make_motions

from motion_forecast.models import LADDER, RAW, MotionForecast, inputs


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    df = make_motions()
    make_model, grid = LADDER["text_metadata"]
    pipeline = make_model(**grid[0]).fit(inputs(df), df["label"])
    path = tmp_path_factory.mktemp("serve") / "model"
    # The input example gives the model its schema, which the server enforces.
    mlflow.pyfunc.save_model(
        path, python_model=MotionForecast(pipeline), input_example=df[RAW].head(2)
    )
    return TestClient(scoring_server.init(mlflow.pyfunc.load_model(str(path))))


def test_server_scores_raw_motions(client):
    # As a client would send them: one Motion has no party, which goes out as null.
    records = json.loads(make_motions(4)[RAW].to_json(orient="records"))
    assert records[3]["party"] is None
    response = client.post("/invocations", json={"dataframe_records": records})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 4
    assert all(0 < p < 1 for p in predictions)


def test_server_rejects_a_motion_without_text(client):
    record = json.loads(make_motions(4)[RAW].drop(columns="text").to_json(orient="records"))[0]
    response = client.post("/invocations", json={"dataframe_records": [record]})
    assert response.status_code == 400
