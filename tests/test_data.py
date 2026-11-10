import pytest

from motion_forecast.data import build, clean_html, validate


def record(number, type_, authors, department, proposal=None, text="x"):
    """A raw record from the service, trimmed to the fields build() reads."""
    return {
        "id": int(number.replace(".", "20", 1)),
        "number": number,
        "type": {"key": type_},
        "submission": {"date": "2026-11-03", "council": {"abbreviation": "NR"}},
        "title": "Titel",
        "texts": {"submittedText": text, "statementOfReasons": "<li>Grund</li>"},
        "response": {"date": "2027-01-15", "proposal": {"name": proposal}} if proposal else None,
        "authors": authors,
        "drafts": [
            {"competentAuthorities": [{"leading": True, "authority": {"abbreviation": department}}]}
        ],
        "currentStatus": {"name": "Erledigt"},
    }


def member(party, group="Fraktion S"):
    party = {"shortName": party} if party else None
    return {"assemblyMembers": [{"party": party, "parliamentaryGroup": {"shortName": group}}]}


def organ(kind, name):
    return {"organ": [{"type": {"name": kind}, "shortName": name}]}


def make_rows():
    return [
        record("26.1001", "MOTION", member("CVP"), "EDI", "Ablehnung", "<p>Der&nbsp;Bundesrat</p>"),
        record("26.1002", "POSTULAT", organ("Fraktion", "Fraktion V"), "EVD", "Annahme"),
        record("26.1003", "MOTION", organ("Kommission", "SGK-N"), "VBS", "Annahme (teilweise)"),
        record("26.1004", "MOTION", member("EVP"), "EDI"),
    ]


def test_build_labels_only_clear_yes_or_no():
    df = build(make_rows())
    # Reject, accept, partial acceptance (no label), not yet answered (no label).
    assert df["label"].tolist()[:2] == [0, 1]
    assert df["label"].iloc[2:].isna().all()


def test_build_maps_authors_to_todays_parties():
    df = build(make_rows())
    # CVP member -> Mitte, SVP group -> SVP, a committee, a small party -> Andere.
    assert df["party"].tolist() == ["Mitte", "SVP", "Kommission", "Andere"]


def test_build_falls_back_to_group_for_members_without_party():
    rows = [record("26.1005", "MOTION", member(None, "Fraktion RL"), "EDI", "Annahme")]
    assert build(rows)["party"].tolist() == ["FDP"]


def test_build_uses_todays_department_names():
    assert build(make_rows())["department"].tolist() == ["EDI", "WBF", "VBS", "EDI"]


def test_build_reads_types_texts_and_dates():
    df = build(make_rows())
    assert df["type"].tolist() == ["Motion", "Postulat", "Motion", "Motion"]
    assert df["text"].iloc[0] == "Der Bundesrat"
    assert str(df["submitted"].iloc[0].date()) == "2026-11-03"
    assert df["proposal_date"].isna().tolist() == [False, False, False, True]


def test_clean_html_keeps_only_words():
    assert clean_html("<p>Der Bundesrat&nbsp;wird <b>beauftragt</b></p>") == (
        "Der Bundesrat wird beauftragt"
    )
    assert clean_html(None) == ""


def test_validate_accepts_clean_data():
    validate(build(make_rows()), min_labelled=2)


def test_validate_rejects_duplicates():
    df = build(make_rows()).assign(id=1)
    with pytest.raises(ValueError, match="duplicate"):
        validate(df, min_labelled=0)


def test_validate_rejects_incomplete_download():
    with pytest.raises(ValueError, match="incomplete"):
        validate(build(make_rows()), min_labelled=3)
