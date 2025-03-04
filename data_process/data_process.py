import os
import argparse
import pandas as pd
from typing import Dict, Union, Tuple
import glob
import datasets

from processors.demographics import get_demographics_data, get_demographic_summary
from processors.measurements import get_bmi_data, get_bmi_statistics
from processors.cbc import get_cbc_data, get_cbc_subject_statistics
from processors.outcomes import get_mortality_data

DEFAULT_PATH = '/Users/aashnashah/Desktop/ssh_mount/data/EHRSHOT/meds_omop_ehrshot/'

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process EHR data for ML tasks.")
    parser.add_argument("--raw", type=bool, default=True)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_PATH)
    parser.add_argument("--output_dir", type=str, default='../data')
    args = parser.parse_args()

    return args

def load_dataset(data_path: str) -> pd.DataFrame:
    """Load dataset from Parquet files and convert to a Pandas DataFrame."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} does not exist")
    
    dataset = datasets.Dataset.from_parquet(os.path.join(data_path, 'data/*'))
    return dataset.to_pandas()

def save_csv(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
            output_dir: str, 
            name: str = None) -> None:
    """
    Save DataFrame(s) to CSV file(s).
    """
    if isinstance(data, pd.DataFrame) and name is None:
        raise ValueError("Name parameter is required when saving a single DataFrame")
    
    data_dict = {name: data} if isinstance(data, pd.DataFrame) else data
    
    os.makedirs(output_dir, exist_ok=True)
    
    for df_name, df in data_dict.items():
        filepath = os.path.join(output_dir, f"{df_name}.csv")
        df.to_csv(filepath, index=False)
        print(f"Saved {df_name} to {filepath}")

def get_dataset_statistics(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], name: str = "", stage: str = "") -> None:
    """Print unified statistics for dataset(s)."""
    if stage:
        print(f"\n{stage} Dataset Statistics:")
    
    data_dict = {name or "Dataset": data} if isinstance(data, pd.DataFrame) else data
    
    for df_name, df in data_dict.items():
        print(f"\n{df_name} data:")
        print(f"# Events: {df.shape[0]}")
        print(f"# Unique subjects: {len(df['subject_id'].unique())}")
        print(f"Average # Events per Patient: {df.groupby('subject_id').size().mean():.2f}")

def subset_patient_data(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Merge patient dataframes based on common subject IDs."""
    
    subject_sets = {name: set(df["subject_id"].unique()) for name, df in data_dict.items()}
    common_subjects = set.intersection(*subject_sets.values()) if subject_sets else set()
    
    return {
        name: df[df['subject_id'].isin(common_subjects)]
        for name, df in data_dict.items()
    }

def merge_patient_data(data_dict: Dict[str, pd.DataFrame], time_tolerance: pd.Timedelta = pd.Timedelta("4D")) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Merge patient data both by subject ID and time-based matching."""
    # First get common subjects across all datasets
    subject_sets = {name: set(df["subject_id"].unique()) for name, df in data_dict.items()}
    common_subjects = set.intersection(*subject_sets.values()) if subject_sets else set()
    
    filtered_dict = {
        name: df[df['subject_id'].isin(common_subjects)]
        for name, df in data_dict.items()
    }
    
    # Then do time-based merging for measurements
    time_series_dfs = {
        name: df.copy().sort_values(['time'])
        for name, df in filtered_dict.items() 
        if 'time' in df.columns
    }
    
    if time_series_dfs:
        # Ensure consistent datetime format for all time columns
        for df in time_series_dfs.values():
            # Convert to UTC then remove timezone info
            df['time'] = pd.to_datetime(df['time']).astype('datetime64[s]')
            df.sort_values('time', inplace=True)
            
        merged_df = pd.merge_asof(
            time_series_dfs['cbc_measurements'],
            time_series_dfs['body_measurements'],
            on="time",
            by="subject_id",
            direction="nearest",
        )
        print(merged_df.head())
        
        # Merge demographics
        demographics_df = filtered_dict['demographics'].copy()
        demographics_df['subject_id'] = demographics_df['subject_id'].astype(int)
        final_df = merged_df.merge(demographics_df, on='subject_id', how='left')
        
        # Convert DOB to datetime and remove timezone info for age calculation
        final_df['DOB'] = pd.to_datetime(final_df['DOB']).astype('datetime64[s]')
        final_df['Age'] = (final_df['time'] - final_df['DOB']).dt.days / 365.25
        
    else:
        final_df = pd.DataFrame()
        
    return filtered_dict, final_df

def get_processed_data(name: str, process_fn, args, dependencies=None) -> pd.DataFrame:
    """Process data or load if exists."""
    processed_path = os.path.join(args.output_dir, "processed", f"{name}.csv")
    
    # Load cached data if exists and not raw mode
    if not args.raw and os.path.exists(processed_path):
        print(f"Loading {name} data from {processed_path}")
        df = pd.read_csv(processed_path)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time']).astype('datetime64[s]')
        return df
    
    # Load raw data if needed
    if 'raw_df' not in globals():
        globals()['raw_df'] = load_dataset(args.data_dir)
        raw_df['time'] = pd.to_datetime(raw_df['time']).astype('datetime64[s]')
        get_dataset_statistics(raw_df, name="Raw", stage="Initial")
    
    # Process data
    print(f"\nProcessing {name} data...")
    result = process_fn(raw_df, *dependencies) if dependencies else process_fn(raw_df)
    save_csv(result, os.path.join(args.output_dir, "processed"), name)
    return result

def main():
    args = parse_args()
            
    # Process data in stages
    data_dict = {}
    
    # Base data
    demographics = get_processed_data("demographics", get_demographics_data, args=args)
    get_dataset_statistics(demographics, name="Demographics", stage="Initial")
    
    body_measurements = get_processed_data("body_measurements", get_bmi_data, args=args)
    get_dataset_statistics(body_measurements, name="Body Measurements", stage="Initial")
    
    cbc_measurements = get_processed_data("cbc_measurements", get_cbc_data, args=args, dependencies=[demographics])
    get_dataset_statistics(cbc_measurements, name="CBC Measurements", stage="Initial")
    
    #mortality = get_processed_data("mortality", get_mortality_data, args=args, dependencies=[demographics])
    # Combine all processed data
    data_dict = {
        "demographics": demographics,
        "body_measurements": body_measurements,
        "cbc_measurements": cbc_measurements
    }
    
    # Merge data
    print("\nMerging Data:")
    filtered_data, merged_df = merge_patient_data(data_dict)
    # Save outputs
    save_csv(filtered_data, os.path.join(args.output_dir, "merged"))
    save_csv(merged_df, args.output_dir, name="combined_subject_cbc_events")
    
    get_dataset_statistics(merged_df, name="Combined CBC Events", stage="Final")
    per_subject_stats = get_cbc_subject_statistics(filtered_data["cbc_measurements"])
    save_csv(per_subject_stats, os.path.join("../results/summary_statistics"), name="cbc_subject_statistics")
    
if __name__ == "__main__":
    main()