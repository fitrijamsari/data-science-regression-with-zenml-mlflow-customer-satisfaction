import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s - %(levelname)s] : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Evaluation(ABC):
    """
    Abstract class for defining strategy for models evaluation
    """

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass


class MAE(Evaluation):
    """Evaluation that use Mean Absolute Error (measures the average absolute differences between the predicted and actual values)"""

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the scores for the model

        Args:
            y_true: True label
            y_pred: Predicted label
        Returns:
            mse: float
        """
        try:
            logging.info("Calculating MAE")
            mae = mean_absolute_error(y_true, y_pred)
            logging.info(f"MAE: {mae}")
            return mae
        except Exception as e:
            logging.error(f"Error in calculating MAE: {e}")
            raise e


class MSE(Evaluation):
    """Evaluation that use Mean Squared Error (measures the average squared differences between the predicted and actual values)"""

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the scores for the model

        Args:
            y_true: True label
            y_pred: Predicted label
        Returns:
            mse: float
        """
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE: {e}")
            raise e


class RMSE(Evaluation):
    """Evaluation that use Root Mean Squared Error (the square root of the MSE)"""

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the scores for the model

        Args:
            y_true: True label
            y_pred: Predicted label
        Returns:
            rmse: float
        """
        try:
            logging.info("Calculating RMSE")
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            logging.info(f"RMSE: {rmse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating RMSE: {e}")
            raise e


class R2Score(Evaluation):
    """Evaluation that use R2 Score (represents the proportion of the variance in the dependent variable that is predictable from the independent variables.
    It ranges from 0 to 1, where 1 indicates a perfect fit)"""

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the scores for the model

        Args:
            y_true: True label
            y_pred: Predicted label
        Returns:
            r2: float
        """
        try:
            logging.info("Calculating R2Score")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R2Score: {e}")
            raise e
