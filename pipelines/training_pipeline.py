from zenml import pipeline

from steps.clean_data import clean_data
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_data
from steps.model_train import train_model


@pipeline(enable_cache=False)
def training_pipeline(data_path: str):
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, X_test, y_train, y_test)
    mae, mse, rmse, r2_score = evaluate_model(model, X_test, y_test)
