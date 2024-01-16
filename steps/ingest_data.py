import logging

import pandas as pd
from zenml import step

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s - %(levelname)s] : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)


@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from the data_path

    Args:
        data_path: path to the data
    Return:
        pd.Dataframe
    """

    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(e)
        raise e
