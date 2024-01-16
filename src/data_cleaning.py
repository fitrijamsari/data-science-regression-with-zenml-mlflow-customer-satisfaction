import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from pandas.core.api import DataFrame as DataFrame
from pandas.core.api import Series as Series
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s - %(levelname)s] : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class DataStrategy(ABC):
    """
    Abstract class defining blueprint startegy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes columns which are not required, fills missing values with median average values, and converts the data type to float.
        PENDING: DO A PROPER DATA HANDLING
        """
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
            )
            data["product_weight_g"].fillna(
                data["product_weight_g"].median(), inplace=True
            )
            data["product_length_cm"].fillna(
                data["product_length_cm"].median(), inplace=True
            )
            data["product_height_cm"].fillna(
                data["product_height_cm"].median(), inplace=True
            )
            data["product_width_cm"].fillna(
                data["product_width_cm"].median(), inplace=True
            )
            # write "No review" in review_comment_message column
            data["review_comment_message"].fillna("No review", inplace=True)

            # NOTE: for simplicity, remove all categorical column. Actually we need to convert categorical data into numerical data here.
            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)

            return data
        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise e


class LabelEncoding(DataStrategy):
    """
    Label encoding categorical column to numerical
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            df_cat = []
            lencod = LabelEncoder()
            for col in df_cat:
                data[col] = lencod.fit_transform(data[col])
            logging.info(data.head())
            return data
        except Exception as e:
            logging.error(f"Error in Label Encoding data: {e}")
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Split dataset into train test
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop(
                "review_score", axis=1
            )  # EDIT column read from config file maybe?
            y = data[["review_score"]]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in Data Splitting: {e}")
            raise e


class DataCleaning:
    """
    Data cleaning and split dataset into train-test
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        return self.strategy.handle_data(self.data)


# if __name__ == "__main__":
#     data = pd.read.csv(
#         "/Users/ofotech_fitri/Documents/fitri_github/data-science-customer-satisfaction-with-zenml/data/olist_customers_dataset.csv"
#     )
#     data_cleaning = DataCleaning(data, DataPreProcessStrategy())
#     data_cleaning.handle_data()
