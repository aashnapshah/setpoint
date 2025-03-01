import logging
import os
import pandas as pd
from typing import Dict, List

logger = logging.getLogger(__name__)

# Constants
CBC_ABBREVIATIONS = {
    'HCT': 'hematocrit',
    'HGB': 'hemoglobin',
    'MCH': 'mean corpuscular hemoglobin',
    'MCHC': 'mean corpuscular hemoglobin concentration',
    'MPV': 'mean platelet volume', 
    'PLT': 'platelet count',
    'RBC': 'red cell count',
    'RDW': 'red cell distribution width',
    'WBC': 'white cell count'
}

# get in the format of {code: loinc_code}
LOINC_CODES = {
    'HGB': "LOINC/718-7", 
    'HCT': "LOINC/4544-3", 
    'RBC': "LOINC/789-8", 
    'PLT': "LOINC/777-3", 
    'MCH': "LOINC/785-6", 
    'MCHC': "LOINC/786-4", 
    'MCV': "LOINC/787-2", 
    'RDW': "LOINC/788-0", 
    'WBC': "LOINC/6690-2", 
}
LOINC_CODES = {v: k for k, v in LOINC_CODES.items()}

CBC_REFERENCE_INTERVALS = {
    'HCT': {'F': (36, 46, '%'), 'M': (41, 53, '%')},  
    'HGB': {'F': (12.0, 16.0, 'g/dL'), 'M': (13.5, 17.5, 'g/dL')},  
    'MCH': {'F': (26, 34, 'pg'), 'M': (26, 34, 'pg')},  
    'MCHC': {'F': (31, 37, 'g/dL'), 'M': (31, 37, 'g/dL')},  
    'MPV': {'F': (8.4, 12.0, 'fL'), 'M': (8.4, 12.0, 'fL')},  
    'PLT': {'F': (150, 400, '10³/µL'), 'M': (150, 400, '10³/µL')},  
    'RBC': {'F': (4.0, 5.2, '10⁶/µL'), 'M': (4.5, 5.9, '10⁶/µL')},  
    'RDW': {'F': (11.5, 14.5, '%'), 'M': (11.5, 14.5, '%')},  
    'WBC': {'F': (4.5, 11.0, '10³/µL'), 'M': (4.5, 11.0, '10³/µL')},
    'MCV': {'F': (80, 100, 'fL'), 'M': (80, 100, 'fL')}
}

UNIT_CONVERSIONS = {
    'g/dL': 'g/dL',
    'G/DL': 'g/dL',
    'g/dl': 'g/dL',
    '%': '%',
    'None': 'None',
    'K/uL': 'K/uL',
    'KUL': 'K/uL',
    'x10E3/uL': 'K/uL',  
    '10x3/uL': 'K/uL',   
    'pg': 'pg',
    'PG': 'pg',
    'fL': 'fL',
    'FL': 'fL',
    'fl': 'fL',
    'Thousand/uL': 'K/uL',
    'Million/uL': 'Million/uL',  
    'MUL': 'Million/uL',
    'MIL/uL':'Million/uL',
    '10*6/uL': 'Million/uL',
    '10x6/uL': 'Million/uL',
    'M/uL': 'Million/uL',
    'x10E6/uL': 'Million/uL',
}

def get_cbc_data(df: pd.DataFrame, demographic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process CBC test results from the dataset.
    """
    cbc_df = extract_tests(df)
    cbc_df = standardize_units(cbc_df)
    cbc_df = filter_cbc(cbc_df)
    cbc_df = get_in_reference_interval(cbc_df, demographic_df)
    
    # print the number of records removed and print number of unique subjects before
    print(f"After filtering: {len(df)} ({df.subject_id.nunique()} subjects) -> {len(cbc_df)} records ({cbc_df.subject_id.nunique()} subjects)")
    return cbc_df

def extract_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract CBC test results from the dataset.
    """
    cbc_df = df[
        (df['table'] == 'measurement') & 
        (df['code'].isin(LOINC_CODES.keys()))
    ].copy()
    cbc_df['code'] = cbc_df['code'].replace(LOINC_CODES)
    return cbc_df

def standardize_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize units for CBC measurements.
    """
    df['unit'] = df['unit'].replace(UNIT_CONVERSIONS)
    check_units(df)
    return df

def check_units(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Check for unit consistency in lab tests.
    """
    unit_issues = {}
    for code in df['code'].unique():
        units = df[df['code'] == code]['unit'].unique()
        if len(units) > 1:
            unit_issues[code] = units.tolist()
            logger.warning(f"{code} has multiple units: {units}")
    
    return unit_issues

def filter_cbc(df: pd.DataFrame, min_tests: int = 5, min_days_between: int = 0) -> pd.DataFrame:
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
    
    #print(f"Aggregated duplicates: {len(df)} ({df.subject_id.nunique()} subjects) -> {len(agg_df)} records ({agg_df.subject_id.nunique()} subjects) for {agg_df.code.nunique()} tests")
    
    # Filter by minimum number of tests
    if min_tests > 1:
        filtered_df = agg_df.groupby(['subject_id', 'code']).filter(lambda x: len(x) >= min_tests)
        
        if min_days_between > 0:
            filtered_df = filter_by_time_interval(filtered_df, min_days_between)
        
        # print the number of records removed and print number of unique subjects before and after filtering
        #print(f"After filtering for multiple measurements: {len(agg_df)} ({agg_df.subject_id.nunique()} subjects) -> {len(filtered_df)} records ({filtered_df.subject_id.nunique()} subjects)")
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
            'num_tests_within_reference': len(group[group['within_reference'] == True]),
            'percentage_tests_within_reference': f"{len(group[group['within_reference'] == True]) / len(group) * 100:.2f}%",
            'days_between_first_and_last': (group['time'].max() - group['time'].min()).days,
            'avg_days_between_tests': f"{time_diffs.mean():.2f}" if not time_diffs.empty else None,
            'min_days_between_tests': f"{time_diffs.min():.2f}" if not time_diffs.empty else None,
            'max_days_between_tests': f"{time_diffs.max():.2f}" if not time_diffs.empty else None
        })
    
    summary_df = df.groupby(['subject_id', 'code']).apply(compute_summary).reset_index().sort_values(by='days_between_first_and_last', ascending=False)
    return summary_df

def get_cbc_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get statistics for CBC tests.
    """
    check_units(df)
    # For each code, print the number of tests, the number of tests within reference interval, and the number of tests outside reference interval, mean and median of numeric_value
    for code in df['code'].unique():
        code_df = df[df['code'] == code]
        print(f"# {code} tests: {len(code_df)}")
        print(f"# {code} tests within reference interval: {len(code_df[code_df['within_reference'] == True])}")
        print(f"# {code} tests outside reference interval: {len(code_df[code_df['within_reference'] == False])}")
        print(f"# {code} mean: {code_df['numeric_value'].mean()}")
        print(f"# {code} median: {code_df['numeric_value'].median()}")
        print(f"# {code} min: {code_df['numeric_value'].min()}")
        print(f"# {code} max: {code_df['numeric_value'].max()}")
    return 


def get_in_reference_interval(df: pd.DataFrame, demographic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag subjects whose CBC values are within the reference interval.
    """
    # Merge the demographic data with the CBC data
    merged_df = pd.merge(df, demographic_df[['subject_id', 'Gender']], on='subject_id', how='left')
    merged_df['within_reference'] = merged_df.apply(lambda row: is_in_reference_interval(row), axis=1)
    merged_df.drop(columns='Gender', inplace=True)
    return merged_df

def is_in_reference_interval(row: pd.Series) -> bool:
    """
    Check if a CBC value is within the reference interval.
    """
    # Get the reference interval for the test
    reference_interval = CBC_REFERENCE_INTERVALS[row['code']]   
    
    # Check if the value is within the reference interval
    if row['Gender'] == 'F':
        return row['numeric_value'] >= reference_interval['F'][0] and row['numeric_value'] <= reference_interval['F'][1]
    else:
        return row['numeric_value'] >= reference_interval['M'][0] and row['numeric_value'] <= reference_interval['M'][1]
    
def get_cbc_overall_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate detailed statistics for each CBC test type from summary data.
    """
    
    # Get statistics by test code
    test_stats = {}
    for code in df['code'].unique():
        code_df = df[df['code'] == code]
        # Add number of subjects within reference interval
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
