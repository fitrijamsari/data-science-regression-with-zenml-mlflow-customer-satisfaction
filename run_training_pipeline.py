import logging

from zenml.client import Client
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

from pipelines.training_pipeline import training_pipeline

# from zenml.post_execution import get_pipeline, get_pipelines


logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s - %(levelname)s] : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    # print(Client().active_stack.experiment_tracker.get_tracking_uri())
    # Run the pipeline
    training_pipeline(
        data_path="/Users/ofotech_fitri/Documents/fitri_github/data-science-customer-satisfaction-with-zenml/data/olist_customers_dataset.csv"
    )

    logging.info(
        "Run the following on terminal: \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`experiment. Here you'll also be able to compare the multiple runs."
    )

    # last_run = Client().get_pipeline("training_pipeline")
    # print(last_run)
    # trainer_step = last_run.get_steps(step="evaluate_model")
    # print(trainer_step)
    # tracking_url = trainer_step.run_metadata.get("experiment_tracker_url")
    # print(tracking_url.value)
