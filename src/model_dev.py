import logging
from abc import ABC, abstractmethod

import xgboost as xgb
from sklearn.linear_model import LinearRegression

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s - %(levelname)s] : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Model(ABC):
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        pass


class LinerRegressionModel(Model):
    """
    LinearRegressionModel that implements the Model interface.
    """

    def train(self, X_train, y_train, **kwargs):
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("LinearRegression Model Training Completed")
            return reg
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise e


class XGBoostModel(Model):
    """XGBoostModel that implements the Model interface"""

    def train(self, X_train, y_train, **kwargs):
        try:
            reg = xgb.XGBRegressor(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("XGBoost Model Training Completed")
            return reg
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise e
