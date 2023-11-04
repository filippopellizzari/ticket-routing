import logging

import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES_LIST = [
    "state",
    "consumer_disputed",
    "product",
    "company_response",
    "submitted_via",
    "timely",
    "consumer_consent_provided",
    "forwarding_days",
    "text",
]
TARGET = "category"

TRAIN_PATH = "data/processed/train.parquet"
TEST_PATH = "data/processed/test.parquet"


def main() -> None:
    """Select features and split train test"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    df = pd.read_parquet("data/interim/complaints_labeled.parquet")
    df = df.set_index("complaint_id")
    logging.info("select features")
    df = df[FEATURES_LIST + [TARGET]]
    logging.info("train test split")
    train, test = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=0
    )
    logging.info(f"train size: {train.shape}")
    logging.info(f"test size: {test.shape}")
    logging.info("output")
    train.to_parquet(TRAIN_PATH)
    test.to_parquet(TEST_PATH)


if __name__ == "__main__":
    main()
