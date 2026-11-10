"""Download Motionen and Postulate from the Swiss parliament's open data, clean and label them.

    python -m motion_forecast.data

Writes data/motions.parquet: one row per Motion or Postulat with its text, its author's
party and, once the Federal Council has answered, the label (1 = it proposes to accept).
"""

import html
import re
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

DATA_PATH = Path("data/motions.parquet")

BASE_URL = "https://ws.parlament.ch/odata.svc/"
PAGE = 1000  # the service returns at most 1000 rows per request
# German records only: texts submitted in French or Italian are translated, so one
# language covers everything.
SCOPE = "Language eq 'DE' and (BusinessType eq 5 or BusinessType eq 6)"
TYPES = {5: "Motion", 6: "Postulat"}
AUTHOR_ROLE = 7  # BusinessRole.Role "Urheber(-in)"; 3 would be the co-signers

# FederalCouncilProposal codes. The service switched to a shorter code set in 2023.
# Everything else has no clear yes or no and gets no label: turning a Motion into a
# Postulat, writing it off as fulfilled, partial acceptance, proposals of the Büro.
ACCEPT = {1, 5, 18, 19, 44}
REJECT = {3, 6, 20, 21, 45}
LABELS = {code: 1 for code in ACCEPT} | {code: 0 for code in REJECT}

# The service gives each member's last party, not the party at submission time. Mapping
# merged parties to today's party makes old and new Motionen comparable: CVP and BDP
# merged into Die Mitte in 2021, FDP and the Liberals into FDP-Liberale in 2009.
PARTIES = {
    "SVP": "SVP",
    "SP": "SP",
    "FDP-Liberale": "FDP",
    "Lib*": "FDP",
    "LDP": "FDP",
    "CVP": "Mitte",
    "M-E": "Mitte",
    "BDP": "Mitte",
    "GRÜNE": "Grüne",
    "glp": "GLP",
}
# Motionen can also come from a parliamentary group (Fraktion), by its number.
GROUPS = {1: "FDP", 2: "SP", 3: "Mitte", 4: "SVP", 6: "Grüne", 136: "Mitte", 137: "GLP"}
GROUPS |= {139: "Mitte", 140: "Mitte"}  # older names of the Mitte group

# A complete download has about 17,600 labelled rows; far fewer means it was cut short.
MIN_LABELLED = 15_000

# Retry only errors that may pass on their own (rate limit, server trouble).
RETRY = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
HTTP = requests.Session()
HTTP.mount("https://", HTTPAdapter(max_retries=RETRY))


def fetch(entity: str, select: list[str], where: str) -> pd.DataFrame:
    """All rows of one OData entity, page by page."""
    rows = []
    while True:
        params = {
            # Always select columns: whole records with all their texts make the
            # service slow enough to time out.
            "$select": ",".join(select),
            "$filter": where,
            "$orderby": "ID",  # a fixed order, so pages neither overlap nor skip rows
            "$skip": len(rows),
            "$top": PAGE,
            "$format": "json",
        }
        response = HTTP.get(BASE_URL + entity, params=params, timeout=300)
        response.raise_for_status()
        page = response.json()["d"]
        # Some entities wrap the rows in {"results": [...]}, others send the list itself.
        page = page["results"] if isinstance(page, dict) else page
        rows += page
        if len(page) < PAGE:
            return pd.DataFrame(rows, columns=select)


def parse_date(values: pd.Series) -> pd.Series:
    # OData v2 sends dates as "/Date(1793664000000)/": milliseconds since 1970, at
    # midnight UTC of the Swiss date (here 2026-11-03).
    ms = values.str.extract(r"/Date\((-?\d+)\)/", expand=False).astype(float)
    return pd.to_datetime(ms, unit="ms")


def clean_html(text: str | None) -> str:
    """The texts are HTML snippets (<p>, <li>, &nbsp;); the models only need the words."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return re.sub(r"\s+", " ", text).strip()


def author_party(member_party: str | None, group: float, committee: float) -> str:
    """Today's party of whoever submitted it: a member, a parliamentary group or a committee."""
    if isinstance(member_party, str):
        return PARTIES.get(member_party, "Andere")
    if pd.notna(group):
        return GROUPS.get(int(group), "Andere")
    if pd.notna(committee):
        return "Kommission"
    return "Andere"


def build(business: pd.DataFrame, authors: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """One clean row per Motion or Postulat, from the three raw tables."""
    # The service sends the odd record twice (24.3206 in November 2026).
    business = business.drop_duplicates("ID").reset_index(drop=True)
    authors = authors.drop_duplicates("BusinessNumber")
    member_parties = members.set_index("ID")["PartyAbbreviation"]
    authors = authors.assign(member_party=authors["MemberCouncilNumber"].map(member_parties))
    parties = pd.Series(
        [
            author_party(*row)
            for row in authors[["member_party", "ParlGroupNumber", "CommitteeNumber"]].itertuples(
                index=False
            )
        ],
        index=authors["BusinessNumber"],
    )

    return pd.DataFrame(
        {
            "id": business["ID"],
            "number": business["BusinessShortNumber"],
            # Known when it is submitted: the model's inputs.
            "type": business["BusinessType"].map(TYPES),
            "submitted": parse_date(business["SubmissionDate"]),
            "council": business["SubmissionCouncilAbbreviation"],
            "party": business["ID"].map(parties).fillna("Andere"),
            # Set on submission, before the Federal Council answers.
            "department": business["ResponsibleDepartmentAbbreviation"],
            "title": business["Title"],
            "text": business["SubmittedText"].map(clean_html),
            "reason": business["ReasonText"].map(clean_html),
            # Known only after the Federal Council's answer: the outcome, never an input.
            "status": business["BusinessStatusText"],
            "proposal": business["FederalCouncilProposal"],
            "proposal_date": parse_date(business["FederalCouncilProposalDate"]),
            "label": business["FederalCouncilProposal"].map(LABELS),
        }
    )


def validate(df: pd.DataFrame, min_labelled: int = MIN_LABELLED) -> None:
    """Fail loudly instead of silently storing bad data."""
    problems = []
    if df["id"].duplicated().any():
        problems.append("duplicate ids")
    for column in ["id", "type", "submitted", "title"]:
        if df[column].isna().any():
            problems.append(f"{column} missing")
    if df["label"].notna().sum() < min_labelled:
        problems.append(f"fewer than {min_labelled} labelled rows: download incomplete?")
    if problems:
        raise ValueError("; ".join(problems))


def update(path: Path = DATA_PATH) -> None:
    business = fetch(
        "Business",
        [
            "ID",
            "BusinessShortNumber",
            "BusinessType",
            "SubmissionDate",
            "SubmissionCouncilAbbreviation",
            # Not the topic tags (TagNames): they make each page ~30x slower, and the
            # department and the text carry the topic anyway.
            "ResponsibleDepartmentAbbreviation",
            "Title",
            "SubmittedText",
            "ReasonText",
            "BusinessStatusText",
            "FederalCouncilProposal",
            "FederalCouncilProposalDate",
        ],
        SCOPE,
    )
    authors = fetch(
        "BusinessRole",
        ["ID", "BusinessNumber", "MemberCouncilNumber", "ParlGroupNumber", "CommitteeNumber"],
        f"{SCOPE} and Role eq {AUTHOR_ROLE}",
    )
    members = fetch("MemberCouncil", ["ID", "PartyAbbreviation"], "Language eq 'DE'")

    df = build(business, authors, members)
    validate(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

    labelled = df.dropna(subset="label")
    print(f"{len(df)} Motionen and Postulate, {len(labelled)} labelled")
    print(f"accepted: {labelled['label'].mean():.0%}")
    print(df["party"].value_counts().to_string())


def load(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


if __name__ == "__main__":
    update()
