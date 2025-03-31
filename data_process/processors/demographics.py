import os
import logging
import pandas as pd
from datetime import datetime
from typing import Set, Tuple, Dict, List

logger = logging.getLogger(__name__)

DEMOGRAPHIC_FIELDS = ['subject_id', 'gender', 'dob', 'race', 'ethnicity']
AGE_RANGE = {'min': 18, 'max': 89}
RACE_MAP = {
    '1': 'American Indian or Alaska Native',
    '2': 'Asian',
    '3': 'Black or African American',
    '4': 'Native Hawaiian or Other Pacific Islander',
    '5': 'White'
}

GENDER_MAP = {
    'M': 'Male',
    'F': 'Female'
}

def process_demographics(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    Extract and filter demographic data.
    """    
    demo_df = get_demographics(df)
    dob_df = get_dob(df)
    demo_df = pivot_and_filter_demographics([demo_df, dob_df])
    observation_period_df = get_observation_period(df, path)
    
    demo_df = demo_df.merge(observation_period_df, on='subject_id', how='left')
    
    return demo_df

def get_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Extract basic demographic information (gender, race, ethnicity)."""
    demo_df = df[df['table'] == 'person'][['subject_id', 'code', 'time']]
    demo_df['text_value'] = demo_df['code'].str.split('/').str[1]
    demo_df['code'] = demo_df['code'].str.split('/').str[0].str.lower()
    demo_df = demo_df[demo_df['code'].isin(DEMOGRAPHIC_FIELDS)]
    return demo_df

def clean_demographics(df: pd.DataFrame) -> pd.DataFrame:
    race_mask = df['code'] == 'race'
    df.loc[race_mask, 'text_value'] = df.loc[race_mask, 'text_value'].map(RACE_MAP)
    
    sex_mask = df['code'] == 'gender'
    df.loc[sex_mask, 'text_value'] = df.loc[sex_mask, 'text_value'].map(GENDER_MAP)
    return df

# AGE NEEDS TO BE FIXED ->  THERE SHOULD BE AN AGE FOR EACH VISIT / TIME POINT
def get_dob(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process age information from birth dates.
    """
    age_df = df[df['code'] == 'MEDS_BIRTH'][['subject_id', 'code', 'time']]
    age_df['code'] = 'dob'    #age_df['Age (Today)'] = (pd.Timestamp.today() - pd.to_datetime(age_df['time'])).dt.days // 365
    age_df['text_value'] = age_df['time']
    return age_df[['subject_id', 'code', 'text_value']]

# give a list of dataframes and return a single dataframe
def pivot_and_filter_demographics(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Pivot demographics into wide format and filter by age range.
    """
    demo_df = pd.concat(dfs)
    demo_df = (demo_df.pivot(index='subject_id', 
                            columns='code', 
                            values='text_value')
                     .reset_index()
                     .dropna(subset=DEMOGRAPHIC_FIELDS))
    
    return demo_df[DEMOGRAPHIC_FIELDS]

def get_zip_code(df: pd.DataFrame) -> dict:
    """
    Generate zip code summary statistics.
    """
    pass

def get_observation_period(raw_df: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    Get observation period, preferring EHR_record observation dates over the observation_period table.
    """
    fallback_dates = (pd.read_csv(os.path.join(path, "tables/observation_period.csv"))
                     .rename(columns={'person_id': 'subject_id'}))
    
    # Filter raw data for record observations
    filtered_df = raw_df[
        (raw_df['table'] != 'death') & 
        (raw_df['visit_id'].notna()) &
        (raw_df['provider_id'].notna()) &
        (raw_df['table'] != 'person') & 
        (~raw_df['time'].isna()) 
    ]
    
    # Get record first and last observations
    record_dates = (filtered_df.groupby('subject_id')
                    .agg(
                        record_start=('time', 'min'),
                        record_end=('time', 'max')
                    )
                    .reset_index())
    print(record_dates.head())
    
    observation_period_df = fallback_dates.merge(record_dates, on='subject_id', how='outer')
    print(observation_period_df.head())
    
    date_columns = ['observation_period_start_DATE', 'observation_period_end_DATE', 'record_start', 'record_end']
    for col in date_columns:
        observation_period_df[col] = pd.to_datetime(observation_period_df[col]).dt.date
    
    start_fallbacks = observation_period_df['observation_period_start_DATE'].isna().sum()
    end_fallbacks = observation_period_df['observation_period_end_DATE'].isna().sum()
    
    # Calculate differences where we have both dates
    start_diff = (pd.to_datetime(observation_period_df['record_start']) - pd.to_datetime(observation_period_df['observation_period_start_DATE'])).dt.days
    end_diff = (pd.to_datetime(observation_period_df['record_end']) - pd.to_datetime(observation_period_df['observation_period_end_DATE'])).dt.days
    
    print(f"\nObservation Period Statistics:")
    print(f"Using fallback start dates for {start_fallbacks} subjects ({start_fallbacks/len(observation_period_df)*100:.1f}%)")
    print(f"Using fallback end dates for {end_fallbacks} subjects ({end_fallbacks/len(observation_period_df)*100:.1f}%)")
    print("\nDifferences between record and fallback dates (in days):")
    print(f"Start date difference (record - fallback): mean={start_diff.mean():.1f}, median={start_diff.median():.1f}")
    print(f"End date difference (record - fallback): mean={end_diff.mean():.1f}, median={end_diff.median():.1f}")
    
    # Prefer fallback dates over record dates
    observation_period_df['observation_period_start_time'] = (
        observation_period_df['observation_period_start_DATE'].fillna(
            observation_period_df['record_start']
        )
    )
    observation_period_df['observation_period_end_time'] = (
        observation_period_df['observation_period_end_DATE'].fillna(
            observation_period_df['record_end']
        )
    )
    
    observation_period_df = observation_period_df[[
        'subject_id', 
        'observation_period_start_time', 
        'observation_period_end_time'
    ]]
    
    return observation_period_df

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
    
    ethnicity_counts = df['Ethnicity'].value_counts()
    ethnicity_stats = {
        'Ethnicity': {
            ethnicity: f"{count:,d} ({100 * count / len(df):.1f}%)"
            for ethnicity, count in ethnicity_counts.items()
        }
    }
    
    summary_stats = {
        'Total Events': f"{len(df):,d}",
        'Total Unique Subjects': f"{df['subject_id'].nunique():,d}",
        'Average Events per Subject': f"{len(df) / df['subject_id'].nunique():.1f}"
    }
    
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


