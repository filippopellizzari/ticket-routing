import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class BaselineModel:
    def __init__(self, categorical_cols: list[str], text_cols: list[str]):
        self.categorical_cols = categorical_cols
        self.text_cols = text_cols

    def _get_pipeline(self) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "one-hot",
                    OneHotEncoder(handle_unknown="ignore"),
                    self.categorical_cols,
                ),
                ("tfidf", TfidfVectorizer(), "text"),
            ],
            remainder="passthrough",
        )
        pipeline = Pipeline(
            [("preprocessor", preprocessor), ("classifier", LogisticRegression())]
        )
        return pipeline

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        pipe = self._get_pipeline()
        pipe.fit(X_train, list(y_train))
        return pipe
