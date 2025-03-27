import os
import pandas as pd
import sys
import pickle
from pathlib import Path
from typing import Tuple
from datetime import datetime

#add path path to data folder using sys.path
sys.path.append('../data_process/')

from data_process import load_dataset
from processors.cbc import CBC_REFERENCE_INTERVALS
from processors.outcomes import process_diagnosis, load_diagnosis_dict, get_first_diagnosis


DEFAULT_PATH = '/Users/aashnashah/Desktop/ssh_mount/data/EHRSHOT/meds_omop_ehrshot/'
raw_df = load_dataset(DEFAULT_PATH)

df = raw_df[raw_df['code'].str.startswith('SNOMED') |
            raw_df['code'].str.startswith('ICD10') | 
            raw_df['code'].str.startswith('ICD9')]
df.time = pd.to_datetime(df.time).dt.date

# For each subject_id and time, aggregate the codes in list and make a new column with the number of codes
aggregated_df = df.groupby(['subject_id', 'time']).agg(
    disease_codes=('code', lambda x: list(set(x))),
    num_codes=('code', 'size')
).reset_index()
print('Aggregated df:')
print(aggregated_df.head())

sample = 115967904
subject_agg_df = aggregated_df[aggregated_df['subject_id'] == sample]
subject_df = df[df['subject_id'] == sample]

print('Subject agg df:')
print(subject_agg_df.head())
print(subject_agg_df.shape)

print('Subject df:')
print(subject_df.head())
print(subject_df.shape)
# find if all the codes were aggregated properly

# Does the total number of codes match the number of codes in the subject_df?
print('Total number of codes per day (sum):')
print(subject_agg_df['num_codes'].sum())

print('Total number of codes in subject_df (raw count):')
print(len(subject_df['code']))

print('Total number of unique codes in subject_df:')
print(subject_df['code'].nunique())

print('Total number of unique codes across all days in aggregated_df:')
print(len(set([code for codes in subject_agg_df['disease_codes'] for code in codes])))

# This should be True
print('\nDo the raw counts match?')
print(len(subject_df['code']) == subject_agg_df['num_codes'].sum())

# This should also be True
print('\nDo the unique codes match?')
print(subject_df['code'].nunique() == len(set([code for codes in subject_agg_df['disease_codes'] for code in codes])))

# load the icd codes
path = '/Users/aashnashah/Desktop/ssh_mount/SETPOINT/data/icd_codes.pkl'
codes_dict = load_diagnosis_dict(path)
disease_code_sets = {disease: set(codes.keys()) for disease, codes in codes_dict.items()}
print(disease_code_sets['chronic_heart_disease'])

# Create binary columns for each disease
for disease in codes_dict.keys():
    # For each row, check if any code in disease_codes matches any code in the disease list
    aggregated_df[disease] = aggregated_df['disease_codes'].apply(
        lambda code_list: any(
            any(str(code).startswith(disease_code) for disease_code in disease_code_sets[disease])
            for code in code_list
        ) if isinstance(code_list, list) else False
    )


# Now your subject_agg_df will also have these columns
subject_agg_df = aggregated_df[aggregated_df['subject_id'] == sample]

# Print to see the results
print("\nDiseases found for sample patient:")
disease_cols = list(codes_dict.keys())
print(subject_agg_df[['time', 'disease_codes'] + ['chronic_heart_disease']].head())

# You can also see when each disease was first diagnosed
first_diagnosis = subject_agg_df[disease_cols].any()
print("\nDiseases ever diagnosed:")
print(first_diagnosis[first_diagnosis].index.tolist())

# After creating the disease columns in aggregated_df
first_diagnoses = get_first_diagnosis(aggregated_df)

# Look at the results
print("\nFirst diagnosis dates for each disease:")
print(first_diagnoses.head(10))

# For your sample patient
print("\nFirst diagnosis dates for sample patient:")
print(first_diagnoses[first_diagnoses['subject_id'] == sample])



