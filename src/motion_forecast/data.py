"""Download Motionen and Postulate from the Swiss parliament's web service, clean and label them.

    PARL_API_KEY=... python -m motion_forecast.data

Writes data/motions.parquet: one row per Motion or Postulat with its text, its author's
party and, once the Federal Council has answered, the label (1 = it proposes to accept).
"""

import html
import os
import re
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

DATA_PATH = Path("data/motions.parquet")

BASE_URL = "https://ws.parl.ch/api/v1/"
PAGE = 1000  # the service returns at most 1000 rows per request
TYPES = {"MOTION": "Motion", "POSTULAT": "Postulat"}
# Only the parts of each record we use; whole records also carry co-signers and history.
FIELDS = "id,number,type,submission,title,texts,response,authors,drafts,currentStatus"

# The Federal Council proposes "Annahme" (accept) or "Ablehnung" (reject). Anything else
# has no clear yes or no and gets no label: partial acceptance, or no proposal at all.
LABELS = {"Annahme": 1, "Ablehnung": 0}

# The service gives each member's last party, not the party at submission time. Mapping
# merged parties to today's party makes old and new Motionen comparable: CVP and BDP
# merged into Die Mitte in 2021, FDP and the Liberals into FDP-Liberale in 2009.
PARTIES = {
    "SVP": "SVP",
    "SP": "SP",
    "FDP-Liberale": "FDP",
    "LDP": "FDP",
    "CVP": "Mitte",
    "CVPO": "Mitte",  # the Upper Valais section of the CVP
    "M-E": "Mitte",
    "BDP": "Mitte",
    "GRÜNE": "Grüne",
    "GB": "Grüne",  # Grünes Bündnis Bern and BastA! Basel: local sections of the Greens
    "BastA": "Grüne",
    "glp": "GLP",
}
# Motionen can also come from a parliamentary group (Fraktion), and members without a
# party still sit in one.
GROUPS = {
    "Fraktion V": "SVP",
    "Fraktion S": "SP",
    "Fraktion RL": "FDP",
    "Fraktion L": "FDP",
    "Fraktion M-E": "Mitte",
    "Fraktion BD": "Mitte",
    "Fraktion G": "Grüne",
    "Fraktion GL": "GLP",
}
# Departments keep the name they had at the time; these were renamed since.
DEPARTMENTS = {"EVD": "WBF", "EVED": "UVEK", "EMD": "VBS"}

# A complete download has about 17,600 labelled rows; far fewer means it was cut short.
MIN_LABELLED = 15_000

# Retry only errors that may pass on their own (rate limit, server trouble).
RETRY = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
HTTP = requests.Session()
HTTP.mount("https://", HTTPAdapter(max_retries=RETRY))


def fetch(business_type: str) -> list[dict]:
    """All records of one business type, page by page."""
    rows = []
    while True:
        params = {
            "q": f'eq(type.key,"{business_type}")',
            "sort": "id:asc",  # a fixed order, so pages neither overlap nor skip rows
            "page": len(rows) // PAGE,
            "size": PAGE,
            "lang": "de",  # texts submitted in French or Italian come translated
            "include": FIELDS,
        }
        headers = {"X-API-KEY": os.environ["PARL_API_KEY"]}
        response = HTTP.get(BASE_URL + "businesses", params=params, headers=headers, timeout=300)
        response.raise_for_status()
        body = response.json()
        rows += body["content"]
        if not body["page"]["hasNext"]:
            return rows


def clean_html(text: str | None) -> str:
    """The texts are HTML snippets (<p>, <li>, &nbsp;); the models only need the words."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return re.sub(r"\s+", " ", text).strip()


def author_party(authors: dict) -> str:
    """Today's party of whoever submitted it: a member, a parliamentary group or a committee."""
    members = authors.get("assemblyMembers") or []
    organs = authors.get("organ") or []
    if members:
        party = (members[0].get("party") or {}).get("shortName")
        group = (members[0].get("parliamentaryGroup") or {}).get("shortName")
        if party in (None, "-"):  # no party: fall back to the group
            return GROUPS.get(group, "Andere")
        return PARTIES.get(party, "Andere")
    if organs and organs[0]["type"]["name"] == "Fraktion":
        return GROUPS.get(organs[0]["shortName"], "Andere")
    if organs:
        return "Kommission"  # a committee, subcommittee or the council's bureau
    return "Andere"


def department(drafts: list[dict]) -> str | None:
    """The department that leads the Federal Council's answer, by today's name."""
    for authority in drafts[0].get("competentAuthorities") or []:
        if authority["leading"]:
            name = authority["authority"]["abbreviation"]
            return DEPARTMENTS.get(name, name)
    return None


def build(rows: list[dict]) -> pd.DataFrame:
    """One clean row per Motion or Postulat, from the raw records."""
    # Flatten the nested records: {"response": {"date": ...}} becomes "response.date".
    raw = pd.json_normalize(rows)
    proposal = raw["response.proposal.name"]
    return pd.DataFrame(
        {
            "id": raw["id"],
            "number": raw["number"],
            # Known when it is submitted: the model's inputs.
            "type": raw["type.key"].map(TYPES),
            "submitted": pd.to_datetime(raw["submission.date"]),
            "council": raw["submission.council.abbreviation"],
            "party": [author_party(row["authors"]) for row in rows],
            # Set soon after submission, before the Federal Council answers.
            "department": [department(row["drafts"]) for row in rows],
            "title": raw["title"],
            "text": raw["texts.submittedText"].map(clean_html),
            "reason": raw["texts.statementOfReasons"].map(clean_html),
            # Known only after the Federal Council's answer: the outcome, never an input.
            "status": raw["currentStatus.name"],
            "proposal": proposal,
            "proposal_date": pd.to_datetime(raw["response.date"]),
            "label": proposal.map(LABELS),
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
    df = build(fetch("MOTION") + fetch("POSTULAT"))
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
