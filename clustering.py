import logging

import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src import data_cleaning, feature_engineering

INPUT_PATH = "data/interim/complaints.parquet"
OUTPUT_PATH = "data/interim/complaints_labeled.parquet"

categories_map = {
    0: "others",
    1: "mortgage",
    2: "bank_account",
    3: "credit_card",
    4: "theft_dispute",
}


def main() -> None:
    """Prepare data, TF IDF text columns, perform KMeans clustering and output dataset with cluster labels."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Data Preparation")
    df = pd.read_parquet(INPUT_PATH)
    df = data_cleaning.rename_columns(df)
    df = data_cleaning.drop_not_useful_cols(df)
    df = feature_engineering.get_forwarding_days(df)
    logging.info("Join text")
    df["text"] = (
        df["product"]
        .str.cat(df["sub_product"], sep=" ")
        .str.cat(df["issue"], sep=" ")
        .str.cat(df["complaint_what_happened"], sep=" ")
    )

    df = df.dropna(subset=["text"])
    logging.info("TF IDF")
    vectorizer = TfidfVectorizer(stop_words="english")
    X_tfidf = vectorizer.fit_transform(df["text"])
    logging.info("Clustering")
    kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(X_tfidf)
    logging.info("Ouput")
    df["label"] = kmeans.labels_
    df["category"] = df["label"].replace(categories_map)
    df.to_parquet(OUTPUT_PATH)


if __name__ == "__main__":
    main()
