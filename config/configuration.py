from zenml.steps import BaseParameters


class ModelNameConfig(BaseParameters):
    """Model name config"""

    # model_name: str = "LinearRegression"
    model_name: str = "XGBoost"
