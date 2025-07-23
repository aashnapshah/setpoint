import pandas as pd
import numpy as np
import pickle as pkl
import sys
sys.path.append('../')
from process.config import *
from functools import reduce

def make_mem_end_csv(df):
    obs_table = pd.read_csv("../data/raw/tables_csv/observation.csv").rename(columns={'person_id': 'subject_id'})
    obs_pd_table = pd.read_csv("../data/raw/tables_csv/observation_period.csv").rename(columns={'person_id': 'subject_id'})
    death_table = pd.read_csv("../data/raw/tables_csv/death.csv").rename(columns={'person_id': 'subject_id'})

    # last record overall
    last_ehr_record = df.groupby('subject_id', as_index=False)['time'].max()
    last_ehr_record.rename(columns={"time":"last_ehr_time"}, inplace=True)

    # last non-null visit id
    filtered_df = df[df['visit_id'].notna()]
    last_visit_id = filtered_df.groupby('subject_id', as_index=False)['time'].max()
    last_visit_id.rename(columns={"time":"last_visit_id"}, inplace=True)

    # obs pd end date
    obs_pd_end_date = obs_pd_table[['subject_id', 'observation_period_end_DATE']]
    obs_pd_end_date['observation_period_end_DATE'] = pd.to_datetime(obs_pd_end_date['observation_period_end_DATE'])
    last_obs = obs_table.groupby('subject_id', as_index=False)['observation_DATETIME'].max()

    # date death
    death_dt = death_table[['subject_id', 'death_DATE']]
    death_dt['death_DATE'] = pd.to_datetime(death_dt['death_DATE'])

    dfs = [obs_pd_end_date, last_obs, last_visit_id, last_ehr_record, death_dt]
    # Merge all on 'subject_id', keeping all subject_ids and filling missing with NaN
    censorship_df = reduce(lambda left, right: pd.merge(left, right, on='subject_id', how='outer'), dfs)
    return censorship_df

def disease_onset_table(df, disease='t2d'):
    pattern = ICD_CODES_REGEX[disease]
    disease_mask = df['code'].str.contains(pattern, regex=True, na=False)
    disease_df = df[disease_mask]
    earliest_disease = disease_df.groupby('subject_id', as_index=False)['time'].min()
    return earliest_disease 

# Get (right) censor dates from the mem_end table
def censorship_algo(df):
    df = df.copy()

    # Ensure all date columns are datetime
    date_cols = ['death_DATE', 'observation_period_end_DATE', 'observation_DATETIME', 'last_visit_id', 'last_ehr_time']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Mortality status: 1 if death_DATE is present, else 0
    df['mortality_status'] = df['death_DATE'].notna().astype(int)

    # Censorship date
    df['censorship_date'] = df['death_DATE']

    # Where death_DATE is null, compute fallback censorship date
    mask_alive = df['mortality_status'] == 0

    # Get row-wise max of the three date columns
    df_alive = df.loc[mask_alive]
    max_dates = df_alive[['observation_period_end_DATE', 'observation_DATETIME', 'last_visit_id']].max(axis=1)

    # Fill with max_dates first, then with last_ehr_time if all are null
    censorship_fallback = max_dates.combine_first(df_alive['last_ehr_time'])

    df.loc[mask_alive, 'censorship_date'] = censorship_fallback

    return df[['subject_id', 'censorship_date']]

def main():
    raw_df = pd.read_parquet('../data/raw/meds_omop_ehrshot/data/')

    # Get mem_end dates and convert to final censor dates and save both
    mem_end_dates = make_mem_end_csv(raw_df)
    mem_end_dates.to_csv("../data/processed/mem_end_dates.csv")
    censored_df = censorship_algo(mem_end_dates)
    censored_df.to_csv("../data/processed/censorship_df.csv", index=False)

    diseases = ['t2d', 'ckd', 'osteo', 'mace', 'af', 'hypertension', 'hyperlipidemia', 'nafld|nash']
    disease_dfs = []

    # Get disease onset dates
    for disease in diseases:
        print(f"DISEASE: {disease}")
        df = disease_onset_table(raw_df, disease)
        df = df.rename(columns={'time': f'{disease}_date'})
        disease_dfs.append(df)

    disease_df = reduce(lambda left, right: pd.merge(left, right, on='subject_id', how='outer'), disease_dfs)

    # Get mortality dates
    mortality_df = pd.read_csv('../data/raw/tables_csv/death.csv')[['person_id', 'death_DATE']].rename(columns={'death_DATE': 'mortality_date', 'person_id': 'subject_id'})

    # Merge all dataframes
    outcomes = disease_df.merge(censored_df, on='subject_id', how='outer').merge(mortality_df, on='subject_id', how='outer')
    # save the outcomes dataframe
    outcomes.to_csv('../data/processed/outcomes_df.csv')

if __name__=='__main__':
    main()
    