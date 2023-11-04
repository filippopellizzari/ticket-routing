import json
import logging

import pandas as pd

JSON_PATH = "data/raw/complaints-2021-05-14_08_16_.json"
PARQUET_PATH = "data/interim/complaints.parquet"


def convert_json_to_pandas_df(json_path: str):
    """Read json and convert to pandas df."""
    f = open(json_path)
    data = json.load(f)
    df = pd.json_normalize(data)
    return df


def main() -> None:
    """Import json with tickets, convert to pandas dataframe and output parquet file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Read json")
    df = convert_json_to_pandas_df(JSON_PATH)
    logging.info(f"Dataset size:{df.shape}")
    logging.info("Output parquet")
    df.to_parquet(PARQUET_PATH)


if __name__ == "__main__":
    main()
