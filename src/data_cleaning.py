import pandas as pd

NOT_USEFUL_COLS = ["index", "type", "id", "score", "company", "company_public_response"]


def drop_not_useful_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop not useful columns."""
    df = df.drop(columns=NOT_USEFUL_COLS)
    return df
