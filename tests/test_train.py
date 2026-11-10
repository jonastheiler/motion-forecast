import pandas as pd
import pytest

from motion_forecast.models import LADDER, inputs
from motion_forecast.train import periods


def make_motions(n: int = 40) -> pd.DataFrame:
    accepted = [i % 4 == 0 for i in range(n)]
    return pd.DataFrame(
        {
            "number": [f"x.{i}" for i in range(n)],
            "type": ["Motion", "Postulat"] * (n // 2),
            "council": ["NR"] * n,
            "party": ["SP", "SVP", "FDP", None] * (n // 4),
            "department": ["EDI", "UVEK"] * (n // 2),
            "title": ["Titel"] * n,
            "text": ["Bericht prüfen" if a else "Gesetz ändern" for a in accepted],
            "reason": ["Begründung"] * n,
            "label": [float(a) for a in accepted],
            "submitted": pd.Timestamp("2010-01-01"),
            "proposal_date": pd.Timestamp("2010-03-01"),
        }
    )


def test_periods_only_train_on_answers_known_at_the_time():
    df = make_motions(4)
    df["submitted"] = pd.to_datetime(["2018-06-01", "2018-12-01", "2019-06-01", "2023-06-01"])
    # The December 2018 Motion is answered in 2019: too late for training, too early
    # for validation.
    df["proposal_date"] = pd.to_datetime(["2018-08-01", "2019-02-01", "2019-08-01", "2023-08-01"])
    split = periods(df)

    assert split["train"]["number"].tolist() == ["x.0"]
    assert split["validation"]["number"].tolist() == ["x.2"]
    assert split["train_validation"]["number"].tolist() == ["x.0", "x.1", "x.2"]
    assert split["test"]["number"].tolist() == ["x.3"]


def test_inputs_combine_texts_and_fill_missing_metadata():
    x = inputs(make_motions(4))
    assert x["words"].iloc[0] == "Titel. Bericht prüfen Begründung"
    assert x["party"].iloc[3] == "unknown"


@pytest.mark.parametrize("stage", list(LADDER))
def test_every_stage_learns_probabilities(stage):
    make_model, grid = LADDER[stage]
    df = make_motions()
    model = make_model(**grid[0]).fit(inputs(df), df["label"])
    p = model.predict_proba(inputs(df))[:, 1]
    assert ((p > 0) & (p < 1)).all()
