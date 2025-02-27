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

def process_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize measurements and compute BMI.
    """
    # Process weight and height measurements
    weight_df = process_weight(df)
    height_df = process_height(df)
    merged_df = merge_measurements(weight_df, height_df)
    bmi_df = calculate_bmi(merged_df)
    
    # Combine with existing BMI measurements
    final_bmi = combine_with_existing_bmi(df, bmi_df)
    
    # Filter invalid BMI values
    final_bmi = filter_bmi_values(final_bmi)
    
    logger.info(f"BMI processing complete. Records: {len(final_bmi)}")
    return final_bmi

# def process_weight(df: pd.DataFrame) -> pd.DataFrame:
#     """Process and standardize weight measurements."""
#     weight_df = df[df['code'] == LOINC_CODES['WEIGHT']].copy()
#     weight_df = weight_df.apply(infer_and_convert_weight, axis=1)
#     return weight_df

def process_height(df: pd.DataFrame) -> pd.DataFrame:
    """Process and standardize height measurements."""
    height_df = df[df['code'] == LOINC_CODES['HEIGHT']].copy()
    height_df['numeric_value'] *= 0.0254 # convert to meters
    height_df['unit'] = 'meters'
    return height_df

def process_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Process and standardize weight measurements."""
    weight_df = df[df['code'] == LOINC_CODES['WEIGHT']].copy()
    
    def infer_and_convert_weight(row: pd.Series) -> pd.Series:
        if pd.isna(row['unit']):
                if row['numeric_value'] > 1000: 
                    row['unit'] = 'ounces'
                elif row['numeric_value'] > 100:  
                    row['unit'] = 'lbs'
                else:
                    row['unit'] = 'kg'
            
        if row['unit'] == 'ounces':
            row['numeric_value'] *= 0.0283495
        elif row['unit'] == 'lbs':
            row['numeric_value'] *= 0.453592

        row['unit'] = 'kg'
        return row
    
    weight_df = weight_df.apply(infer_and_convert_weight, axis=1)
    return weight_df


def merge_measurements(weight_df: pd.DataFrame, height_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge weight and height measurements.
    """
    merged_df = pd.merge(
        weight_df[['subject_id', 'time', 'numeric_value', 'visit_id']], 
        height_df[['subject_id', 'time', 'numeric_value', 'visit_id']], 
        on=['subject_id', 'time', 'visit_id'],
        how='left',
        suffixes=('_weight', '_height')
    )
    return merged_df

def calculate_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate BMI from height and weight measurements."""
    df['BMI_computed'] = df.apply(
        lambda row: row['numeric_value_weight'] / (row['numeric_value_height'] ** 2) 
        if pd.notna(row['numeric_value_weight']) and pd.notna(row['numeric_value_height']) 
        else None, 
        axis=1
    )
    return df.dropna(subset=['BMI_computed'])

def combine_with_existing_bmi(original_df: pd.DataFrame, computed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine computed BMI with existing BMI measurements.
    """
    # Get existing BMI measurements
    existing_bmi = original_df[original_df['code'] == LOINC_CODES['BMI']].copy()
    existing_bmi = existing_bmi.rename(columns={'numeric_value': 'BMI_existing'})
    
    # Merge computed and existing BMI
    merged_bmi = pd.merge(
        computed_df[['subject_id', 'time', 'visit_id', 'BMI_computed']], 
        existing_bmi[['subject_id', 'time', 'visit_id', 'BMI_existing']], 
        on=['subject_id', 'time', 'visit_id'],
        how='outer'
    )
    
    merged_bmi['final_bmi'] = merged_bmi['BMI_existing'].fillna(merged_bmi['BMI_computed'])
    merged_bmi['BMI_category'] = pd.cut(merged_bmi['final_bmi'], 
                                        bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, 100], 
                                        labels=['Underweight', 'Normal', 'Overweight', 'Obesity', 'Severe Obesity', 
                                                'Morbid Obesity'])
    return merged_bmi

def filter_bmi_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out invalid BMI values.
    """
    # Remove physiologically impossible BMI values
    valid_bmi = df[(df['final_bmi'] >= 10) & (df['final_bmi'] <= 100)]
    
    # Log filtering results
    filtered_count = len(df) - len(valid_bmi)
    if filtered_count > 0:
        logger.warning(f"Removed {filtered_count} invalid BMI values")
        
    return valid_bmi

def get_bmi_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate summary statistics for BMI values.
    """
    stats = {
        'mean_bmi': df['final_bmi'].mean(),
        'median_bmi': df['final_bmi'].median(),
        'std_bmi': df['final_bmi'].std(),
        'min_bmi': df['final_bmi'].min(),
        'max_bmi': df['final_bmi'].max(),
        'total_measurements': len(df),
        'unique_subjects': df['subject_id'].nunique()
    }
    return stats 