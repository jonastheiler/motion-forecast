import numpy as np
import pandas as pd
import pytest

from motion_forecast.forecast import COLUMNS, forecast_new, open_motions, score_answered

TODAY = pd.Timestamp("2026-11-13")


def make_motions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "number": ["26.1", "26.2", "25.3", "26.4"],
            "type": "Motion",
            "council": "NR",
            "party": "SP",
            "department": "EDI",
            "title": ["A", "B", "C", "D"],
            "text": "x",
            "reason": "y",
            # Open, answered, open but too old to be answered, withdrawn.
            "submitted": pd.to_datetime(["2026-10-01", "2026-09-01", "2025-03-01", "2026-10-01"]),
            "proposal_date": pd.to_datetime([None, "2026-11-01", None, None]),
            "status": ["Eingereicht", "Erledigt", "Eingereicht", "Erledigt"],
            "label": [np.nan, 1.0, np.nan, np.nan],
        }
    )


class Constant:
    """Stands in for the champion: forecasts 30% for every Motion."""

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), 0.3)


def test_only_recent_unanswered_motions_are_open():
    assert open_motions(make_motions(), TODAY)["id"].tolist() == [1]


def test_each_motion_is_forecast_once():
    df, none = make_motions(), pd.DataFrame(columns=COLUMNS)
    first = forecast_new(df, none, Constant(), "3", TODAY)
    assert first["id"].tolist() == [1]
    assert first["model_version"].tolist() == ["3"]
    assert forecast_new(df, first, Constant(), "3", TODAY).empty


def test_scores_only_answered_forecasts():
    forecasts = pd.DataFrame({"id": [1, 2], "p_accept": [0.3, 0.8]})
    metrics = score_answered(make_motions(), forecasts)
    # Only Motion 2 is answered: accepted, forecast at 80%.
    assert metrics["answered"] == 1
    assert metrics["live_log_loss"] == pytest.approx(-np.log(0.8))
    assert metrics["actual_rate"] == 1.0


def test_no_answers_yet_scores_nothing():
    forecasts = pd.DataFrame({"id": [1], "p_accept": [0.3]})
    assert score_answered(make_motions(), forecasts) == {"answered": 0}
