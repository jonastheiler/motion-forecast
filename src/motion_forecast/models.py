"""The model ladder: each stage adds one source of information, so the gaps between
stages show what that information is worth.

    majority:       the share of accepted Motionen, the same for every one
    metadata:       who submitted it, where, and which department has to answer
    text:           only the words of the Motion
    text_metadata:  both
    embedding:      the words as one vector from a pretrained language model (embed.py)
    embedding_metadata: the vector and the metadata

MotionForecast wraps a trained stage for deployment.
"""

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

METADATA = ["type", "council", "party", "department"]
# What a caller sends: a Motion as it comes from the parliament, before any features.
RAW = [*METADATA, "title", "text", "reason"]
# The name the deployed model is registered under in MLflow.
MODEL_NAME = "motion-forecast"
WORDS = "words"
EMBEDDING = "embedding"


def inputs(df: pd.DataFrame) -> pd.DataFrame:
    """The model's inputs: only what is known when a Motion is submitted."""
    x = df[METADATA].fillna("unknown")
    # One text per Motion: the title, what it asks for, and why.
    x[WORDS] = df["title"] + ". " + df["text"] + " " + df["reason"]
    if EMBEDDING in df:  # computed from the words by embed.py, once
        x[EMBEDDING] = df[EMBEDDING]
    return x


def one_hot() -> tuple:
    # A party or department that is new in the test period gets all zeros, not an error.
    return ("metadata", OneHotEncoder(handle_unknown="ignore"), METADATA)


def tfidf(ngrams: int) -> tuple:
    # sublinear_tf: a word used ten times is not ten times as telling as one used once.
    # min_df=2 drops words that occur in a single Motion, mostly names and typos.
    vectorizer = TfidfVectorizer(ngram_range=(1, ngrams), min_df=2, sublinear_tf=True)
    return ("words", vectorizer, WORDS)


def stack(vectors: pd.Series) -> np.ndarray:
    """A column with one vector per Motion, as one matrix."""
    return np.vstack(vectors.to_numpy())


def vectors() -> tuple:
    # The vectors are ready-made features: the language model is not trained here.
    return ("embedding", FunctionTransformer(stack), EMBEDDING)


def classifier(C: float) -> LogisticRegression:
    # No class weights: they would push the probabilities up for the rarer "accept" and
    # break calibration, which the log loss measures.
    return LogisticRegression(C=C, max_iter=2000)


def majority() -> Pipeline:
    return make_pipeline(DummyClassifier(strategy="prior"))


def metadata(C: float) -> Pipeline:
    return make_pipeline(ColumnTransformer([one_hot()]), classifier(C))


def text(C: float, ngrams: int) -> Pipeline:
    return make_pipeline(ColumnTransformer([tfidf(ngrams)]), classifier(C))


def text_metadata(C: float, ngrams: int) -> Pipeline:
    return make_pipeline(ColumnTransformer([tfidf(ngrams), one_hot()]), classifier(C))


def embedding(C: float) -> Pipeline:
    return make_pipeline(ColumnTransformer([vectors()]), classifier(C))


def embedding_metadata(C: float) -> Pipeline:
    return make_pipeline(ColumnTransformer([vectors(), one_hot()]), classifier(C))


# Each stage with the hyperparameters to try. C is the inverse regularisation strength.
LADDER = {
    "majority": (majority, [{}]),
    "metadata": (metadata, [{"C": C} for C in [0.1, 1, 10]]),
    "text": (text, [{"C": C, "ngrams": n} for C in [1, 3, 10] for n in [1, 2]]),
    "text_metadata": (text_metadata, [{"C": C, "ngrams": n} for C in [1, 3, 10] for n in [1, 2]]),
    "embedding": (embedding, [{"C": C} for C in [0.1, 0.3, 1, 3]]),
    "embedding_metadata": (embedding_metadata, [{"C": C} for C in [0.1, 0.3, 1, 3]]),
}


class MotionForecast(mlflow.pyfunc.PythonModel):
    """The deployable model: a raw Motion in, its probability of acceptance out.

    The pipeline alone expects the features that inputs() builds. Packing inputs() with
    it means a caller cannot build them differently from how the model was trained.
    """

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        return self.pipeline.predict_proba(inputs(model_input))[:, 1]
