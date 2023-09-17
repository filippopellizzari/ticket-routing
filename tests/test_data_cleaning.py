import pandas as pd
from pandas.testing import assert_frame_equal

from src import data_cleaning


def test_rename_columns_remove_prefix_underscore():
    df = pd.DataFrame({"_my_var": [1, 2], "my_var_2": [4, 5], "._my_var_3": [7, 8]})
    expected_df = pd.DataFrame(
        {"my_var": [1, 2], "my_var_2": [4, 5], "._my_var_3": [7, 8]}
    )

    actual_df = data_cleaning.rename_columns(df)

    assert_frame_equal(expected_df, actual_df)


def test_rename_columns_remove_prefix_source():
    df = pd.DataFrame(
        {
            "source._my_var": [1, 2],
            "_source.my_var_2": [4, 5],
            "source_my_var_3": [7, 8],
        }
    )
    expected_df = pd.DataFrame(
        {"_my_var": [1, 2], "my_var_2": [4, 5], "source_my_var_3": [7, 8]}
    )

    actual_df = data_cleaning.rename_columns(df)

    assert_frame_equal(expected_df, actual_df)
