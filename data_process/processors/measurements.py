import logging
import pandas as pd
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# Constants
LOINC_CODES = {
    'WEIGHT': "LOINC/29463-7",
    'HEIGHT': "LOINC/8302-2",
    'BMI': "LOINC/39156-5"
}

def process_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize measurements and compute BMI.
    """
    # Process weight and height measurements
    weight_df = process_weight(df)
    height_df = process_height(df)
    bmi_df = calculate_bmi(weight_df, height_df)
    agg_bmi = aggregate_bmi(df, bmi_df)
    agg_bmi = filter_bmi(agg_bmi)
    #agg_bmi['time'] = pd.to_datetime(agg_bmi['time'], format='mixed', errors='ignore').dt.date
    return agg_bmi

def process_height(df: pd.DataFrame) -> pd.DataFrame:
    """Process and standardize height measurements."""
    height_df = df[df['code'] == LOINC_CODES['HEIGHT']].copy()
    height_df['adjusted_numeric_value'] = height_df['numeric_value']
    def infer_and_convert_height(row: pd.Series) -> pd.Series:
        if row['numeric_value'] > 90:
            row['unit'] = 'cm'
        elif row['numeric_value'] > 2.5:
            row['unit'] = 'inches'
        else:
            row['unit'] = 'meters'
        
        # Convert to meters
        if row['unit'] == 'cm':
            row['adjusted_numeric_value'] *= 0.01
        elif row['unit'] == 'inches':
            row['adjusted_numeric_value'] *= 0.0254
        
        row['unit'] = 'meters'
        return row
    
    height_df = height_df.apply(infer_and_convert_height, axis=1)
    return height_df

def process_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Process and standardize weight measurements."""
    weight_df = df[df['code'] == LOINC_CODES['WEIGHT']].copy()
    weight_df['adjusted_numeric_value'] = weight_df['numeric_value']
    def infer_and_convert_weight(row: pd.Series) -> pd.Series:
        if row['numeric_value'] > 1000: 
            row['unit'] = 'ounces'
        elif row['numeric_value'] > 100:  
            row['unit'] = 'lbs'
        else:
            row['unit'] = 'kg'
    
        if row['unit'] == 'ounces':
            row['adjusted_numeric_value'] *= 0.0283495
        elif row['unit'] == 'lbs':
            row['adjusted_numeric_value'] *= 0.453592
            
        row['unit'] = 'kg'
        return row
    
    weight_df = weight_df.apply(infer_and_convert_weight, axis=1)
    return weight_df

def calculate_bmi(weight_df: pd.DataFrame, height_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate BMI from height and weight measurements."""
    df = pd.merge(
        weight_df[['subject_id', 'time', 'numeric_value', 'adjusted_numeric_value']], 
        height_df[['subject_id', 'time', 'numeric_value', 'adjusted_numeric_value']], 
        on=['subject_id', 'time'],
        how='left',
        suffixes=('_weight', '_height')
    )
    
    df['BMI_computed'] = df.apply(
        lambda row: row['adjusted_numeric_value_weight'] / (row['adjusted_numeric_value_height'] ** 2) 
        if pd.notna(row['adjusted_numeric_value_weight']) and pd.notna(row['adjusted_numeric_value_height']) 
        else None, 
        axis=1
    )
    return df.dropna(subset=['BMI_computed'])

def aggregate_bmi(df: pd.DataFrame, computed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine computed BMI with existing BMI measurements.
    """
    # Get existing BMI measurements
    existing_bmi = df[df['code'] == LOINC_CODES['BMI']].copy()
    existing_bmi = existing_bmi.rename(columns={'numeric_value': 'BMI_existing'})
    
    # Merge computed and existing BMI
    merged_bmi = pd.merge(
        computed_df[['subject_id', 'time', 'numeric_value_weight', 'numeric_value_height',
                     'adjusted_numeric_value_weight', 'adjusted_numeric_value_height', 'BMI_computed']], 
        existing_bmi[['subject_id', 'time', 'BMI_existing']], 
        on=['subject_id', 'time'],
        how='outer'
    )
    print(f"Calculated BMI: {len(merged_bmi.dropna(subset=['BMI_computed']))} ({merged_bmi.dropna(subset=['BMI_computed']).subject_id.nunique()} subjects)")
    print(f"Extracted BMI: {len(merged_bmi.dropna(subset=['BMI_existing']))} ({merged_bmi.dropna(subset=['BMI_existing']).subject_id.nunique()} subjects)")
    print(f'Total Subjects with BMI: {len(merged_bmi)} ({merged_bmi.subject_id.nunique()} subjects)')
    
    merged_bmi['BMI'] = merged_bmi['BMI_existing'].fillna(merged_bmi['BMI_computed'])
    merged_bmi['time'] = pd.to_datetime(merged_bmi['time'], format='mixed', errors='ignore').dt.date    
    merged_bmi = merged_bmi.groupby(['subject_id', 'time']).agg({'BMI': 'mean'}).reset_index()
    print(merged_bmi.head())

    merged_bmi['BMI_category'] = pd.cut(merged_bmi['BMI'], 
                                        bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, 100], 
                                        labels=['Underweight', 'Normal', 
                                                'Overweight', 'Obesity', 
                                                'Severe Obesity', 'Morbid Obesity'])
    
    return merged_bmi 

def filter_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out invalid BMI values.
    """
    valid_bmi = df[(df['BMI'] >= 5) & (df['BMI'] <= 100)]

    mean_bmi = valid_bmi['BMI'].mean()
    std = valid_bmi['BMI'].std()
    valid_bmi = valid_bmi[(valid_bmi['BMI'] >= mean_bmi - 6 * std) & (valid_bmi['BMI'] <= mean_bmi + 6 * std)]
    
    filtered_count = len(df) - len(valid_bmi)
    print(f"Mean BMI: {mean_bmi}, Std: {std}")
    print(f"Removed {filtered_count} invalid BMI values. New min: {valid_bmi['BMI'].min()}, max: {valid_bmi['BMI'].max()}")
        
    return valid_bmi

def get_bmi_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate summary statistics for BMI values.
    """
    stats = {
        'mean_bmi': df['BMI'].mean(),
        'median_bmi': df['BMI'].median(),
        'std_bmi': df['BMI'].std(),
        'min_bmi': df['BMI'].min(),
        'max_bmi': df['BMI'].max(),
        'total_measurements': len(df),
        'unique_subjects': df['subject_id'].nunique()
    }
    return stats 