import pandas as pd
from pandas.testing import assert_frame_equal

from src import feature_engineering


def test_get_forwarding_days():
    df = pd.DataFrame(
        {
            "date_received": [
                "2017-09-13T12:00:00-05:00",
                "2017-09-13T12:00:00-05:00",
                "2017-09-13T12:00:00-05:00",
            ],
            "date_sent_to_company": [
                "2017-09-13T12:00:00-05:00",
                "2017-09-18T12:00:00-05:00",
                "2017-09-18T11:00:00-05:00",
            ],
        }
    )
    expected_df = pd.DataFrame(
        {
            "date_received": [
                "2017-09-13T12:00:00-05:00",
                "2017-09-13T12:00:00-05:00",
                "2017-09-13T12:00:00-05:00",
            ],
            "date_sent_to_company": [
                "2017-09-13T12:00:00-05:00",
                "2017-09-18T12:00:00-05:00",
                "2017-09-18T11:00:00-05:00",
            ],
            "forwarding_days": [0, 5, 4],
        }
    )

    actual_df = feature_engineering.get_forwarding_days(df)

    assert_frame_equal(expected_df, actual_df)
