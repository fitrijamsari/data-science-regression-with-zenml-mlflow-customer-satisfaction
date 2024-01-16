import logging
from typing import Tuple

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client

from src.evaluation import MAE, MSE, RMSE, R2Score

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s - %(levelname)s] : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def evaluate_model(
    model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.DataFrame
) -> Tuple[
    Annotated[float, "mae"],
    Annotated[float, "mse"],
    Annotated[float, "rmse"],
    Annotated[float, "r2_score"],
]:
    """
    Evaluates the model on test dataset

    Args:
        model: model name
        x_test: Test data
        y_test: Test label
    Return:
        mae : float
        mse : float
        rmse : float
        r2_score : float
    """
    try:
        y_pred = model.predict(X_test)

        mae_class = MAE()
        mae = mae_class.calculate_scores(y_test, y_pred)
        mlflow.log_metric("mae", mae)

        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, y_pred)
        mlflow.log_metric("mse", mse)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, y_pred)
        mlflow.log_metric("rmse", rmse)

        r2_class = R2Score()
        r2_score = r2_class.calculate_scores(y_test, y_pred)
        mlflow.log_metric("r2_score", r2_score)

        return mae, mse, rmse, r2_score

    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e
