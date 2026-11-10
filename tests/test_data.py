import pandas as pd
import pytest

from motion_forecast.data import build, clean_html, parse_date, validate


def make_tables():
    business = pd.DataFrame(
        {
            "ID": [20261001, 20261002, 20261003, 20261004],
            "BusinessShortNumber": ["26.1001", "26.1002", "26.1003", "26.1004"],
            "BusinessType": [5, 6, 5, 5],
            "SubmissionDate": ["/Date(1793664000000)/"] * 4,
            "SubmissionCouncilAbbreviation": ["NR", "SR", "NR", "NR"],
            "ResponsibleDepartmentAbbreviation": ["EDI", "UVEK", "VBS", "EDI"],
            "Title": ["A", "B", "C", "D"],
            "SubmittedText": ["<p>Der Bundesrat&nbsp;wird beauftragt</p>", None, "x", "y"],
            "ReasonText": [None, "<li>Grund</li>", "x", "y"],
            "BusinessStatusText": ["Erledigt", "Erledigt", "Erledigt", "Eingereicht"],
            # 45 = new code for reject, 18 = old code for accept, 2 = turn into a Postulat
            "FederalCouncilProposal": [45, 18, 2, None],
            "FederalCouncilProposalDate": ["/Date(1793664000000)/", None, None, None],
        }
    )
    authors = pd.DataFrame(
        {
            "ID": ["a", "b", "c", "d"],
            "BusinessNumber": [20261001, 20261002, 20261003, 20261004],
            "MemberCouncilNumber": [1.0, None, None, 2.0],
            "ParlGroupNumber": [None, 4.0, None, None],
            "CommitteeNumber": [None, None, 12.0, None],
        }
    )
    members = pd.DataFrame({"ID": [1, 2], "PartyAbbreviation": ["CVP", "EVP"]})
    return business, authors, members


def test_build_labels_only_clear_yes_or_no():
    df = build(*make_tables())
    # Reject, accept, "turn into a Postulat" (no label), not yet answered (no label).
    assert df["label"].tolist()[:2] == [0, 1]
    assert df["label"].iloc[2:].isna().all()


def test_build_drops_records_sent_twice():
    business, authors, members = make_tables()
    business = pd.concat([business, business.iloc[[0]]])
    assert build(business, authors, members)["id"].is_unique


def test_build_maps_authors_to_todays_parties():
    df = build(*make_tables())
    # CVP member -> Mitte, SVP group -> SVP, a committee, a small party -> Andere.
    assert df["party"].tolist() == ["Mitte", "SVP", "Kommission", "Andere"]


def test_clean_html_keeps_only_words():
    assert clean_html("<p>Der Bundesrat&nbsp;wird <b>beauftragt</b></p>") == (
        "Der Bundesrat wird beauftragt"
    )
    assert clean_html(None) == ""


def test_parse_date_reads_odata_dates():
    dates = parse_date(pd.Series(["/Date(1793664000000)/", None]))
    assert dates.iloc[0] == pd.Timestamp("2026-11-03")
    assert pd.isna(dates.iloc[1])


def test_validate_accepts_clean_data():
    validate(build(*make_tables()), min_labelled=2)


def test_validate_rejects_duplicates():
    df = build(*make_tables()).assign(id=1)
    with pytest.raises(ValueError, match="duplicate"):
        validate(df, min_labelled=0)


def test_validate_rejects_incomplete_download():
    with pytest.raises(ValueError, match="incomplete"):
        validate(build(*make_tables()), min_labelled=3)
