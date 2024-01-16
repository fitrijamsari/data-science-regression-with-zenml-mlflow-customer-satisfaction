import logging

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client

from config.configuration import ModelNameConfig
from src.model_dev import LinerRegressionModel, XGBoostModel

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s - %(levelname)s] : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    model_config: ModelNameConfig,
) -> RegressorMixin:
    """
    Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
        model_config: ModelNameConfig
    Returns:
        model: RegressorMixin
    """
    model = None

    if model_config.model_name == "LinearRegression":
        mlflow.sklearn.autolog()  # log model, scores in mlflow
        model = LinerRegressionModel()
        trained_model = model.train(X_train, y_train)
    elif model_config.model_name == "XGBoost":
        mlflow.sklearn.autolog()  # log model, scores in mlflow
        model = XGBoostModel()
        trained_model = model.train(X_train, y_train)

        return trained_model
    else:
        raise ValueError(
            f"Model name {model_config.model_name} not supported. Please configure in configuration file"
        )
