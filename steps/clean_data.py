import logging
from typing import Tuple

import pandas as pd
from typing_extensions import Annotated
from zenml import step

from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessStrategy

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s - %(levelname)s] : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@step
def clean_data(
    data: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.DataFrame, "y_train"],
    Annotated[pd.DataFrame, "y_test"],
]:
    """Data cleaning class which preprocesses the data and divides it into train and test data.

    Args:
        data: pd.DataFrame

    Returns:
        X_train: Training Data
        X_test: Test Data
        y_train : Training Labels
        y_test: Test Labels
    """

    try:
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        cleaned_data = data_cleaning.handle_data()
        logging.info("Data Cleaning Completed")

        divide_strategy = DataDivideStrategy()
        data_split = DataCleaning(cleaned_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_split.handle_data()
        logging.info("Data Splitting Completed")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(e)
        raise e
