import pandas as pd


def get_forwarding_days(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the difference in days between when the ticket is received and when is forwarded to the right department."""
    date_sent_to_company_dt = pd.to_datetime(df["date_sent_to_company"])
    date_received_dt = pd.to_datetime(df["date_received"])
    df["forwarding_days"] = (date_sent_to_company_dt - date_received_dt).dt.days
    return df
