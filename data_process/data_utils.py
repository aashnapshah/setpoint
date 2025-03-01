import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Union
from datetime import datetime

# Constants
CBC_LOINC_CODES = [
    "LOINC/718-7",  # Hemoglobin
    "LOINC/4544-3", # Hematocrit
    "LOINC/789-8",  # RBC Count
    "LOINC/777-3",  # Platelet Count
    "LOINC/785-6",  # MCH
    "LOINC/786-4",  # MCHC
    "LOINC/787-2",  # MCV
    "LOINC/788-0"   # RDW
]

UNIT_CONVERSIONS = {
    'g/dL': 'g/dL', 'G/DL': 'g/dL', 'g/dl': 'g/dL',
    '%': '%', 'None': 'None',
    'K/uL': 'K/uL', 'KUL': 'K/uL', 'x10E3/uL': 'K/uL', '10x3/uL': 'K/uL',
    'Thousand/uL': 'K/uL',
    'pg': 'pg', 'PG': 'pg',
    'fL': 'fL', 'FL': 'fL', 'fl': 'fL',
    'Million/uL': 'Million/uL', 'MUL': 'Million/uL', 'MIL/uL': 'Million/uL',
    '10*6/uL': 'Million/uL', '10x6/uL': 'Million/uL', 'M/uL': 'Million/uL',
    'x10E6/uL': 'Million/uL'
}

def infer_and_convert_weight(row: pd.Series) -> pd.Series:
    """Infer and standardize weight units."""
    if pd.isna(row['unit']):
        if row['numeric_value'] > 1000: 
            row['unit'] = 'ounces'
        elif row['numeric_value'] > 100:  
            row['unit'] = 'lbs'
        else:
            row['unit'] = 'kg'
    
    # Convert to kg
    if row['unit'] == 'ounces':
        row['numeric_value'] *= 0.0283495
    elif row['unit'] == 'lbs':
        row['numeric_value'] *= 0.453592
    
    row['unit'] = 'kg'
    return row

def compute_time_stats(group: pd.DataFrame) -> pd.Series:
    """Compute time-based statistics for a group of measurements."""
    time_diffs = group['time'].diff().dropna().dt.total_seconds() / (60 * 60 * 24)
    return pd.Series({
        'num_tests_taken': len(group),
        'time_between_first_and_last': (group['time'].max() - group['time'].min()).days,
        'avg_time_between_tests': f"{time_diffs.mean():.2f}" if not time_diffs.empty else None,
        'min_time_between_tests': f"{time_diffs.min():.2f}" if not time_diffs.empty else None,
        'max_time_between_tests': f"{time_diffs.max():.2f}" if not time_diffs.empty else None
    })

def sample_subjects(df: pd.DataFrame, fraction: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """Randomly sample a fraction of unique subjects."""
    unique_subjects = df['subject_id'].unique()
    sampled_subjects = np.random.choice(
        unique_subjects, 
        size=int(len(unique_subjects) * fraction), 
        replace=False
    )
    return df[df['subject_id'].isin(sampled_subjects)]

def compute_event_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the number of events per subject."""
    event_counts = df['subject_id'].value_counts().reset_index()
    return event_counts.rename(columns={'index': 'subject_id', 'subject_id': 'event_count'}) 