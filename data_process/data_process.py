# from processors.demographics import DemographicsProcessor
# from processors.measurements import MeasurementsProcessor
# from processors.cbc import CBCProcessor
# from processors.outcomes import OutcomesProcessor
import pandas as pd
import datasets
import os

def load_dataset(data_path: str) -> pd.DataFrame:
    """
    Load dataset from Parquet files and convert to a Pandas DataFrame.
    """
    dataset = datasets.Dataset.from_parquet(os.path.join(data_path, 'data/*'))
    return dataset.to_pandas()
