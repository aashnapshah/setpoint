import logging
import pandas as pd
from datetime import datetime
from typing import Set, Tuple, Dict

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
    return demo_df[['subject_id', 'Gender', 'DOB', 'Race', 'Ethnicity']]

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

def get_demographic_summary(df: pd.DataFrame) -> Dict:
    """
    Generate a clean summary of demographic statistics.
    
    Args:
        df: DataFrame with demographic data
        
    Returns:
        Dictionary with formatted demographic statistics
    """
    # Age statistics
    age_stats = {
        'Age (years)': {
            'Mean ± SD': f"{df['Age'].mean():.1f} ± {df['Age'].std():.1f}",
            'Median [IQR]': f"{df['Age'].median():.1f} [{df['Age'].quantile(0.25):.1f}-{df['Age'].quantile(0.75):.1f}]",
            'Range': f"{df['Age'].min():.1f}-{df['Age'].max():.1f}"
        }
    }
    
    # Sex distribution
    sex_counts = df['Gender'].value_counts()
    sex_stats = {
        'Sex': {
            'Female': f"{sex_counts.get('F', 0):,d} ({100 * sex_counts.get('F', 0) / len(df):.1f}%)",
            'Male': f"{sex_counts.get('M', 0):,d} ({100 * sex_counts.get('M', 0) / len(df):.1f}%)"
        }
    }
    
    # Race distribution
    race_counts = df['Race'].value_counts()
    race_stats = {
        'Race': {
            race: f"{count:,d} ({100 * count / len(df):.1f}%)"
            for race, count in race_counts.items()
        }
    }
    
    # Ethnicity distribution
    ethnicity_counts = df['Ethnicity'].value_counts()
    ethnicity_stats = {
        'Ethnicity': {
            ethnicity: f"{count:,d} ({100 * count / len(df):.1f}%)"
            for ethnicity, count in ethnicity_counts.items()
        }
    }
    
    # Total subjects
    summary_stats = {
        'Total Events': f"{len(df):,d}",
        'Total Unique Subjects': f"{df['subject_id'].nunique():,d}",
        'Average Events per Subject': f"{len(df) / df['subject_id'].nunique():.1f}"
    }
    
    # Combine all statistics
    summary = {
        **summary_stats,
        **age_stats,
        **sex_stats,
        **race_stats,
        **ethnicity_stats
    }
    
    return summary

def print_demographic_summary(summary: Dict):
    """Print demographic summary in a formatted way."""
    print("\nDemographic Summary")
    print("==================")
    
    # Print total subjects
    print(f"\nTotal Subjects: {summary['Total Subjects']}")
    
    # Print age statistics
    print("\nAge Statistics")
    print("--------------")
    for stat, value in summary['Age (years)'].items():
        print(f"{stat}: {value}")
    
    # Print sex distribution
    print("\nSex Distribution")
    print("---------------")
    for sex, value in summary['Sex'].items():
        print(f"{sex}: {value}")
    
    # Print race distribution
    print("\nRace Distribution")
    print("----------------")
    for race, value in summary['Race'].items():
        print(f"{race}: {value}")
    
    # Print ethnicity distribution
    print("\nEthnicity Distribution")
    print("---------------------")
    for ethnicity, value in summary['Ethnicity'].items():
        print(f"{ethnicity}: {value}")

def validate_demographics(demo_df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validate demographic data for common issues.
    """
    issues = []
    
    # Check for missing values
    required_columns = ['subject_id', 'Gender', 'DOB', 'Race', 'Ethnicity']
    missing_values = demo_df[required_columns].isnull().sum()
    if missing_values.any():
        issues.append(f"Missing values found: {missing_values[missing_values > 0].to_dict()}")
    
    return len(issues) == 0, issues 


