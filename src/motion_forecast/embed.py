"""Turn each Motion's words into a vector with a pretrained multilingual model.

    python -m motion_forecast.embed

Writes data/embeddings.parquet: one vector per Motion or Postulat. Embedding is slow on a
CPU (~1 hour for all 20k), so only rows missing from the file are embedded, and progress
is saved as it goes: after a new download, only the new Motionen cost time.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from motion_forecast.data import load
from motion_forecast.models import WORDS, inputs

EMBEDDINGS_PATH = Path("data/embeddings.parquet")
# Reads up to 512 tokens: the title, the request and the start of the reasons. Smaller
# models stop at 128 tokens, which would cut most texts after the first paragraph.
MODEL = "intfloat/multilingual-e5-small"
CHUNK = 2000  # rows embedded between two saves


def embed(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    # e5 models are trained with a prefix: "query: " for a text that stands on its own.
    texts = ["query: " + text for text in texts]
    return model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)


def update(path: Path = EMBEDDINGS_PATH) -> None:
    df = load()
    done = load_embeddings(path) if path.exists() else pd.DataFrame(columns=["id", "embedding"])
    todo = df[~df["id"].isin(done["id"])]
    print(f"{len(done)} already embedded, {len(todo)} to go")
    model = SentenceTransformer(MODEL, device="cpu")
    for start in range(0, len(todo), CHUNK):
        chunk = todo.iloc[start : start + CHUNK]
        vectors = embed(model, inputs(chunk)[WORDS].tolist())
        new = pd.DataFrame({"id": chunk["id"].to_numpy(), "embedding": list(vectors)})
        done = pd.concat([done, new], ignore_index=True)
        done.to_parquet(path, index=False)
        print(f"{len(done)} of {len(df)} embedded")


def load_embeddings(path: Path = EMBEDDINGS_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


if __name__ == "__main__":
    update()
