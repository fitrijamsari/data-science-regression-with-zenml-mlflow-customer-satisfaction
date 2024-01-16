import logging

import pandas as pd


def drop_columns(data: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    """
    Drop specified columns from a Pandas DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): Input DataFrame.
    - columns_to_drop (list): List of column names to drop.

    Returns:
    - pd.DataFrame: DataFrame with specified columns dropped.
    """
    try:
        # Ensure that the DataFrame is not modified in place
        dataframe_copy = data.copy()
        dataframe_copy.drop(columns=cols_to_drop, inplace=True, errors="ignore", axis=1)

        return dataframe_copy

    except Exception as e:
        logging.error(f"Error in Preprocessing Data (Drop Columns): {e}")
        raise e


def fillna_values(
    data: pd.DataFrame, cols_to_fill: list, fill_value=None, method=None
) -> pd.DataFrame:
    """
    Fill missing values in a Pandas DataFrame with a specified value or method.

    Parameters:
    - data (pd.DataFrame): Input DataFrame.
    - columns_to_fill (list): List of column names to fill.
    - fill_value: The value to use for filling NaNs.
    - method: Method to use for filling NaN values. Options: 'ffill' (forward fill),
              'bfill' (backward fill), 'mean', 'median', 'mode', etc.

    Returns:
    - pd.DataFrame: DataFrame with missing values filled.
    """
    # Ensure that the DataFrame is not modified in place
    dataframe_copy = data.copy()

    try:
        # Fill NaN values in specified columns based on the specified method or fill_value
        for column in cols_to_fill:
            if method:
                dataframe_copy[column].fillna(method=method, inplace=True)
            elif fill_value is not None:
                dataframe_copy[column].fillna(value=fill_value, inplace=True)
    except Exception as e:
        logging.error(f"Error in Preprocessing Data (Fill Columns): {e}")
        raise e

    return dataframe_copy
