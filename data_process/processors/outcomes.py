import logging
import pandas as pd
import pickle
from typing import Dict, List

def process_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Process outcomes information."""
    mortality_df = get_mortality_date(df)
    return mortality_df

def get_mortality_date(df: pd.DataFrame) -> pd.DataFrame:
    """Process death information."""
    print("Processing death information...")
    death_df = df[df['table'] == 'death'].rename(columns={'time': 'time_of_death'})
    death_df = death_df[['subject_id', 'time_of_death']]

    return death_df[['subject_id', 'time_of_death']]

def process_diagnosis(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Process diagnosis information."""
    print("Processing diagnosis information...")
    codes_dict = load_diagnosis_dict(path)
    df = get_diagnosis_codes(df, codes_dict)
    return df 

def load_diagnosis_dict(path: str) -> dict:
    with open(path, 'rb') as f:
        codes_dict = pickle.load(f)
    for disease in codes_dict:
        print(f"{disease}: {len(codes_dict[disease])}")
    return codes_dict

def code_to_disease(codes_dict: dict) -> dict:
    code_to_disease = {}
    for disease in codes_dict:
        for code in codes_dict[disease]:
            code_to_disease[code] = disease
    return code_to_disease

def get_diagnosis_codes(df: pd.DataFrame, codes_dict: dict) -> pd.DataFrame:
    # Filter the DataFrame to keep only rows with diagnosis codes
    df = df[df['code'].str.startswith('SNOMED') |
            df['code'].str.startswith('ICD10') | 
            df['code'].str.startswith('ICD9')]
    df.time = pd.to_datetime(df.time).dt.date
    
    # For each subject_id and time, aggregate the codes in  list
    aggregated_df = df.groupby(['subject_id', 'time']).agg(
                    disease_codes=('code', lambda x: list(set(x))),
                    num_codes=('code', 'size')
                ).reset_index()

    aggregated_df.rename(columns={'code': 'disease_codes'}, inplace=True)
    disease_code_sets = {disease: set(codes.keys()) for disease, codes in codes_dict.items()}

    for disease in disease_code_sets.keys():
        aggregated_df[disease] = aggregated_df['disease_codes'].apply(
            lambda code_list: any(
                any(str(code).startswith(disease_code) for disease_code in disease_code_sets[disease])
                for code in code_list
            ) if isinstance(code_list, list) else False
        )
    return aggregated_df

def get_first_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the first diagnosis date for each disease for each patient.
    """
    disease_cols = [col for col in df.columns if col not in ['subject_id', 'disease_codes', 'num_codes']]
    first_diagnosis_df = pd.DataFrame(columns=['subject_id'])
    
    for disease in disease_cols:
        disease_df = df[df[disease] == True]
        first_dates = disease_df.groupby('subject_id')['time'].min().reset_index()
        first_dates.columns = ['subject_id', f'{disease}_diagnosis_time']
        
        first_diagnosis_df = pd.merge(
            first_diagnosis_df, 
            first_dates, 
            on='subject_id', 
            how='outer'
        )
    
    short_df = first_diagnosis_df
    long_df = pd.melt(
        first_diagnosis_df,
        id_vars=['subject_id'],
        value_vars=[col for col in first_diagnosis_df.columns if col.endswith('_diagnosis_time')],
        var_name='diagnosis',
        value_name='date'
    )
    
    long_df['diagnosis'] = long_df['diagnosis'].str.replace('_diagnosis_time', '')
    long_df = long_df.dropna(subset=['date'])
    long_df = long_df.sort_values(['subject_id', 'date'])
    
    return short_df, long_df