import pandas as pd
import numpy as np
from typing import Tuple, Optional
import sys, os

sys.path.append( '../')
from process.config import LOINC_CODES, REVERSE_LOINC_CODES, DATAPATH
import logging
import os
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    filename=os.path.join("..", "logs", "preprocess.txt"),
    filemode='a',
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO
)

def log_status(df: pd.DataFrame, message: str) -> None:
    """Log the current status of the DataFrame with a custom message."""
    n_patients, n_measurements = total_ms_pts(df)
    logging.info(f"{message} | {n_patients} Patients | {n_measurements} Measurements")

def filter_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only relevant measurement rows and remove NaNs and non-positive values."""
    df = df[df['table'] == 'measurement']
    # add LOINC/ to the code in loinc_codes
    LOINC_CODES_LIST = [f"LOINC/{code}" for code in REVERSE_LOINC_CODES.keys()]
    df = df[df['code'].isin(LOINC_CODES_LIST)]
    log_status(df, "Removed non-CBC measurements")
    df = df.dropna(subset=['numeric_value'])
    log_status(df, "Removed NaN values")
    df = df[df['numeric_value'] > 0]
    log_status(df, "Removed values <= 0")
    # Fix WBC units
    wbc_mask = (df['code'].isin(LOINC_CODES['WBC'])) & (df['unit'] == "/uL")
    df.loc[wbc_mask, 'numeric_value'] /= 1000
    # Add test name column
    # need to add LOINC/ to the code in loinc_codes
    df['test_name'] = df['code'].str.replace('LOINC/', '').map(REVERSE_LOINC_CODES)
    df['test_name'] = df['test_name'].fillna('NA')
    return df

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove values outside 3 standard deviations for each test."""
    for test_name in LOINC_CODES.keys():
        test_mask = df['test_name'] == test_name
        test_vals = df.loc[test_mask, 'numeric_value']
        mean, std = test_vals.mean(), test_vals.std()
        lower, upper = mean - 3 * std, mean + 3 * std
        mask = ~((test_mask) & ~df['numeric_value'].between(lower, upper))
        df = df[mask]
    log_status(df, "Removed outliers (3 std outside mean)")
    return df

def filter_by_age(df: pd.DataFrame, input_path: str = "../data/table_csv/person.csv") -> pd.DataFrame:
    """Filter out measurements for patients not between 13 and 100 years old at time of test."""
    person_df = pd.read_csv(input_path)
    person_df = person_df.rename(columns={"person_id": "subject_id"})
    df = df.merge(person_df[['subject_id', 'birth_DATETIME']], on='subject_id', how='left')
    df['birth_DATETIME'] = pd.to_datetime(df['birth_DATETIME'])
    df['age'] = (df['time'] - df['birth_DATETIME']).dt.days / 365.25
    df = df[df['age'].between(13, 100)]
    log_status(df, "Enforced 13 < Age < 100 at test")
    return df

def get_sex(df: pd.DataFrame, input_path: str = "../data/table_csv/person.csv") -> pd.DataFrame:
    """Get sex for each patient."""
    person_df = pd.read_csv(input_path)
    person_df = person_df.rename(columns={"person_id": "subject_id"})
    df = df.merge(person_df[['subject_id', 'gender_concept_id']], on='subject_id', how='left')
    # Convert gender_concept_id to binary
    df['gender_concept_id'] = df['gender_concept_id'].map({8507: 0, 8532: 1})
    df['sex'] = df['gender_concept_id'].map(lambda x: 'M' if x == 0 else 'F')
    return df

def total_ms_pts(db):
    return (db['subject_id'].unique().shape[0], db[db['table'] == 'measurement'].shape[0])

def df_search(db, substring):
    mask = db.apply(lambda row: row.astype(str).str.contains(substring, case=False, na=False)).any(axis=1)
    return db[mask]   

def preprocess_pipeline(
    input_path: str = DATAPATH,
    output_path: str = "../data/processed/all_lab_measurements.pkl",
    person_csv_path: str = "../data/raw/tables_csv/person.csv",
) -> pd.DataFrame:
    """Run the full preprocessing pipeline and save the result."""
    df = pd.read_parquet(input_path)
    log_status(df, "Starting point")
    df = filter_measurements(df)
    df = remove_outliers(df)
    df = filter_by_age(df, person_csv_path)
    print(df.head())
    df = get_sex(df, person_csv_path)
    pd.to_pickle(df, output_path)
    df.to_csv(output_path.replace(".pkl", ".csv"), index=False)
    print(f"Final preprocessed data saved to {output_path}")
    return df

if __name__ == '__main__':
    preprocess_pipeline()