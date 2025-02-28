import logging
import pandas as pd
from datetime import datetime
from typing import Set, Tuple

logger = logging.getLogger(__name__)

DEMOGRAPHIC_FIELDS = ['Ethnicity', 'Gender', 'Race']
AGE_RANGE = {'min': 18, 'max': 89}

def get_demographics_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and filter demographic data.
    """    
    demo_df = extract_basic_demographics(df)
    age_df = process_age(df)
    demo_df = pd.concat([demo_df, age_df])
    demo_df = pivot_and_filter_demographics(demo_df)
    return demo_df

def extract_basic_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Extract basic demographic information (gender, race, ethnicity)."""
    demo_df = df[df['table'] == 'person'][['subject_id', 'code', 'time']]
    demo_df['text_value'] = demo_df['code'].str.split('/').str[1]
    demo_df['code'] = demo_df['code'].str.split('/').str[0]
    demo_df = demo_df[demo_df['code'].isin(DEMOGRAPHIC_FIELDS)]
    logger.info(f"Extracted basic demographics for {demo_df['subject_id'].nunique()} subjects")
    return demo_df

# AGE NEEDS TO BE FIXED ->  THERE SHOULD BE AN AGE FOR EACH VISIT / TIME POINT
def process_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process age information from birth dates.
    """
    age_df = df[df['code'] == 'MEDS_BIRTH'][['subject_id', 'code', 'time']]
    age_df['code'] = 'DOB'    #age_df['Age (Today)'] = (pd.Timestamp.today() - pd.to_datetime(age_df['time'])).dt.days // 365
    age_df['text_value'] = age_df['time']
    return age_df

def calculate_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate age from birth date.
    """
    df['time'] = pd.to_datetime(df['time'])
    df['DOB'] = pd.to_datetime(df['dob'])
    df['Age'] = (df['time'] - df['dob']).dt.days // 365
    return df

def pivot_and_filter_demographics(demo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot demographics into wide format and filter by age range.
    """
    demo_df = (demo_df.pivot(index='subject_id', 
                            columns='code', 
                            values='text_value')
                     .reset_index()
                     .dropna(subset=['Gender', 'DOB', 'Race', 'Ethnicity']))
    
    #demo_df = demo_df[(demo_df['Age'].astype(float) >= AGE_RANGE['min']) & 
    #                  (demo_df['Age'].astype(float) <= AGE_RANGE['max'])]
    
    logger.info(f"After filtering: {len(demo_df)} subjects remain")
    return demo_df

def get_care_site_summary(df: pd.DataFrame) -> dict:
    """
    Generate care sites.
    """
    pass

def get_zip_code(df: pd.DataFrame) -> dict:
    """
    Generate zip code summary statistics.
    """
    pass

def get_demographic_summary(demo_df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for demographics.
    """
    summary = {
        'total_subjects': len(demo_df),
        'age_stats': {
            'mean': demo_df['Age'].astype(float).mean(),
            'median': demo_df['Age'].astype(float).median(),
            'std': demo_df['Age'].astype(float).std()
        },
        'gender_distribution': demo_df['Gender'].value_counts().to_dict(),
        'race_distribution': demo_df['Race'].value_counts().to_dict(),
        'ethnicity_distribution': demo_df['Ethnicity'].value_counts().to_dict()
    }
    
    return summary

def validate_demographics(demo_df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validate demographic data for common issues.
    """
    issues = []
    
    # Check for missing values
    required_columns = ['subject_id', 'Gender', 'Age', 'Race', 'Ethnicity']
    missing_values = demo_df[required_columns].isnull().sum()
    if missing_values.any():
        issues.append(f"Missing values found: {missing_values[missing_values > 0].to_dict()}")
    
    return len(issues) == 0, issues 


