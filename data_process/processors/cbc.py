import logging
import pandas as pd
from typing import Dict, List

logger = logging.getLogger(__name__)

# Constants
LOINC_CODES = {
    'HEMOGLOBIN': "LOINC/718-7", # Hemoglobin
    'HEMATOCRIT': "LOINC/4544-3", # Hematocrit
    'RBC_COUNT': "LOINC/789-8", # RBC Count
    'PLATELET': "LOINC/777-3", # Platelet Count
    'MCH': "LOINC/785-6", # Mean Corpuscular Hemoglobin
    'MCHC': "LOINC/786-4", # Mean Corpuscular Hemoglobin Concentration
    'MCV': "LOINC/787-2", # Mean Corpuscular Volume
    'RDW': "LOINC/788-0", # Red Cell Distribution Width
    'WBC_COUNT': "LOINC/6690-2", # White Blood Cell Count
    'NEUTROPHIL': "LOINC/6691-0", # Neutrophil Count
    'LYMPHOCYTE': "LOINC/6692-8", # Lymphocyte Count
    'MONOCYTE': "LOINC/6693-6", # Monocyte Count
    'EOSINOPHIL': "LOINC/6694-4", # Eosinophil Count
    'BASOPHIL': "LOINC/6695-1", # Basophil Count
}

UNIT_CONVERSIONS = {
    # Hemoglobin and MCHC units
    'g/dL': 'g/dL', 'G/DL': 'g/dL', 'g/dl': 'g/dL',
    # Hematocrit and RDW units
    '%': '%',
    # Platelet and WBC units
    'K/uL': 'K/uL', 'KUL': 'K/uL', 
    'x10E3/uL': 'K/uL', '10x3/uL': 'K/uL',
    'Thousand/uL': 'K/uL',
    # MCH units
    'pg': 'pg', 'PG': 'pg',
    # MCV units
    'fL': 'fL', 'FL': 'fL', 'fl': 'fL',
    # RBC units
    'Million/uL': 'Million/uL', 'MUL': 'Million/uL', 
    'MIL/uL': 'Million/uL', 'M/uL': 'Million/uL',
    '10*6/uL': 'Million/uL', '10x6/uL': 'Million/uL',
    'x10E6/uL': 'Million/uL'
}

def process_cbc(df: pd.DataFrame, min_tests: int = 2, min_days_between: int = 0) -> pd.DataFrame:
    """
    Process CBC test results from the dataset.
    """
    cbc_df = extract_tests(df)
    cbc_df = standardize_units(cbc_df)
    cbc_df = filter_cbc(cbc_df, min_tests=2, min_days_between=0)
    return cbc_df

def extract_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract CBC test results from the dataset.
    """
    cbc_df = df[
        (df['table'] == 'measurement') & 
        (df['code'].isin(LOINC_CODES.values()))
    ].copy()
    
    print(f"Extracted {len(cbc_df)} CBC test results")
    return cbc_df

def standardize_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize units for CBC measurements.
    """
    df['unit'] = df['unit'].replace(UNIT_CONVERSIONS)
    return df

def check_units(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Check for unit consistency in lab tests.
    """
    
    # Standardize units first
    df = standardize_units(df)
    
    # Check for multiple units per test
    unit_issues = {}
    for code in df['code'].unique():
        units = df[df['code'] == code]['unit'].unique()
        if len(units) > 1:
            unit_issues[code] = units.tolist()
            logger.warning(f"Test {code} has multiple units: {units}")
    
    return unit_issues

def filter_cbc(df: pd.DataFrame, min_tests: int = 2, min_days_between: int = 0) -> pd.DataFrame:
    """
    Filter CBC tests based on frequency and time intervals.
    """
    df = df.copy()
    df = standardize_units(df)
    df['time'] = pd.to_datetime(df['time'])
    
    # Remove invalid values
    df = df[df['numeric_value'].notna()]
    df = df[df['numeric_value'] > 0]
    
    # Aggregate duplicate measurements
    agg_df = (df.groupby(['subject_id', 'code', 'time', 'unit'])['numeric_value']
              .mean()
              .reset_index())
    
    print(f"Aggregated duplicates: {len(df)} -> {len(agg_df)} records")
    
    # Filter by minimum number of tests
    if min_tests > 1:
        filtered_df = agg_df.groupby(['subject_id', 'code']).filter(lambda x: len(x) >= min_tests)
        
        if min_days_between > 0:
            filtered_df = filter_by_time_interval(filtered_df, min_days_between)
        
        print(f"After filtering for multiple measurements: {len(agg_df)} -> {len(filtered_df)} records")
        return filtered_df
    return filtered_df

def filter_by_time_interval(df: pd.DataFrame, min_days: int) -> pd.DataFrame:
    """
    Filter tests to ensure minimum time interval between measurements.
    """
    def check_interval(group):
        group = group.sort_values('time')
        time_diff = group['time'].diff().dt.total_seconds() / (24 * 3600)  # Convert to days
        return (time_diff >= min_days).all()
    
    return df.groupby(['subject_id', 'code']).filter(check_interval)

def get_cbc_subject_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get subject-level statistics for CBC tests.
    """
    def compute_summary(group):
        time_diffs = group['time'].diff().dropna().dt.total_seconds() / (60 * 60 * 24)  # Convert to days
        return pd.Series({
            'num_tests_taken': len(group),
            'days_between_first_and_last': (group['time'].max() - group['time'].min()).days,
            'avg_days_between_tests': f"{time_diffs.mean():.2f}" if not time_diffs.empty else None,
            'min_days_between_tests': f"{time_diffs.min():.2f}" if not time_diffs.empty else None,
            'max_days_between_tests': f"{time_diffs.max():.2f}" if not time_diffs.empty else None
        })
    
    summary_df = df.groupby(['subject_id', 'code']).apply(compute_summary).reset_index().sort_values(by='days_between_first_and_last', ascending=False)
    return summary_df

def get_cbc_overall_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate detailed statistics for each CBC test type from summary data.
    """
    
    # Get statistics by test code
    test_stats = {}
    for code in df['code'].unique():
        code_df = df[df['code'] == code]
        
        if len(code_df) > 0:
            test_stats[code] = {
                'test_frequency': {
                    'total_subjects': len(code_df),
                'total_tests': code_df['num_tests_taken'].sum(),
                'tests_per_subject': {
                    'mean': code_df['num_tests_taken'].mean(),
                    'median': code_df['num_tests_taken'].median(),
                    'std': code_df['num_tests_taken'].std(),
                    'min': code_df['num_tests_taken'].min(),
                    'max': code_df['num_tests_taken'].max()
                }
            },
            'time_span': {
                'days_between_tests': {
                    'mean': code_df['avg_days_between_tests'].mean(),
                    'median': code_df['avg_days_between_tests'].median(),
                    'std': code_df['avg_days_between_tests'].std(),
                    'min': code_df['min_days_between_tests'].min(),
                    'max': code_df['max_days_between_tests'].max()
                },
                'total_days': {
                    'mean': code_df['days_between_first_and_last'].mean(),
                    'median': code_df['days_between_first_and_last'].median(),
                    'std': code_df['days_between_first_and_last'].std(),
                    'min': code_df['days_between_first_and_last'].min(),
                    'max': code_df['days_between_first_and_last'].max()
                }
            }
        }
    
    # Calculate overall statistics
    overall_stats = {
        'total_subjects': df['subject_id'].nunique(),
        'total_tests': df['num_tests_taken'].sum(),
        'tests_per_subject': {
            'mean': df.groupby('subject_id')['num_tests_taken'].mean().mean(),
            'median': df.groupby('subject_id')['num_tests_taken'].mean().median(),
            'std': df.groupby('subject_id')['num_tests_taken'].mean().std()
        },
        'average_days_between_tests': {
            'mean': df['avg_days_between_tests'].mean(),
            'median': df['avg_days_between_tests'].median(),
            'std': df['avg_days_between_tests'].std()
        },
        'total_days_span': {
            'mean': df['days_between_first_and_last'].mean(),
            'median': df['days_between_first_and_last'].median(),
            'std': df['days_between_first_and_last'].std()
        }
    }
    
    return {
        'by_test': test_stats,
        'overall': overall_stats
    }
