import logging

import joblib
import pandas as pd
from sklearn.metrics import classification_report

from src.baseline_model import BaselineModel

TRAIN_PATH = "data/processed/train.parquet"
TEST_PATH = "data/processed/test.parquet"
CATEGORICAL_COLS = [
    "state",
    "consumer_disputed",
    "product",
    "company_response",
    "submitted_via",
    "timely",
    "consumer_consent_provided",
]
TEXT_COLS = ["text"]
EXPERIMENT_NAME = "baseline"


def get_x_y(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Get X and y from train and test."""
    X_train = train.drop("category", axis=1)
    y_train = train["category"]
    X_test = test.drop("category", axis=1)
    y_test = test["category"]
    return X_train, X_test, y_train, y_test


def main() -> None:
    """Classification Model"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("read dataset")
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    X_train, X_test, y_train, y_test = get_x_y(train, test)
    logging.info("train model")
    model = BaselineModel(categorical_cols=CATEGORICAL_COLS, text_cols=TEXT_COLS)
    pipe = model.train(X_train, y_train)
    logging.info("save model")
    joblib.dump(pipe, f"models/model-{EXPERIMENT_NAME}.pkl")
    logging.info("evauation")
    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)
    report_train = classification_report(y_true=y_train, y_pred=y_pred_train)
    report_test = classification_report(y_true=y_test, y_pred=y_pred_test)
    print(report_train)
    print(report_test)


if __name__ == "__main__":
    main()
