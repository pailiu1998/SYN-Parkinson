#!/usr/bin/env python
"""
Compare files between mixed_scenario_4 and original data folders
"""
import pandas as pd
import numpy as np

def compare_csv_files(file1, file2, name):
    """Compare two CSV files and report differences"""
    print(f"\n{'='*70}")
    print(f"Comparing: {name}")
    print(f"{'='*70}")
    
    # Read files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    print(f"File 1: {file1}")
    print(f"  Rows: {len(df1)}, Columns: {len(df1.columns)}")
    print(f"File 2: {file2}")
    print(f"  Rows: {len(df2)}, Columns: {len(df2.columns)}")
    
    # Check if shapes match
    if df1.shape != df2.shape:
        print(f"⚠ WARNING: Shapes don't match!")
        return
    
    # Check if column names match
    if not all(df1.columns == df2.columns):
        print(f"⚠ WARNING: Column names don't match!")
        print(f"File 1 columns: {list(df1.columns)[:10]}...")
        print(f"File 2 columns: {list(df2.columns)[:10]}...")
        return
    
    # Compare row by row
    different_rows = 0
    same_rows = 0
    different_row_indices = []
    
    for idx in range(len(df1)):
        row1 = df1.iloc[idx]
        row2 = df2.iloc[idx]
        
        # Check if rows are equal (handling NaN values)
        if not row1.equals(row2):
            # Check more carefully, comparing element by element
            differs = False
            for col in df1.columns:
                val1 = row1[col]
                val2 = row2[col]
                
                # Handle NaN comparison
                if pd.isna(val1) and pd.isna(val2):
                    continue
                elif pd.isna(val1) or pd.isna(val2):
                    differs = True
                    break
                elif val1 != val2:
                    # For float values, check if they're very close
                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        if not np.isclose(val1, val2, rtol=1e-9, atol=1e-9):
                            differs = True
                            break
                    else:
                        differs = True
                        break
            
            if differs:
                different_rows += 1
                if len(different_row_indices) < 10:  # Store first 10 different rows
                    different_row_indices.append(idx)
        else:
            same_rows += 1
    
    total_rows = len(df1) - 1  # Excluding header
    percentage_different = (different_rows / total_rows) * 100 if total_rows > 0 else 0
    
    print(f"\nResults:")
    print(f"  Total rows (excluding header): {total_rows}")
    print(f"  Identical rows: {same_rows}")
    print(f"  Different rows: {different_rows}")
    print(f"  Percentage different: {percentage_different:.2f}%")
    
    if different_rows > 0 and len(different_row_indices) > 0:
        print(f"\nFirst few different row indices: {different_row_indices[:10]}")
        
        # Show details of first different row
        first_diff_idx = different_row_indices[0]
        print(f"\nExample - Row {first_diff_idx} differences:")
        row1 = df1.iloc[first_diff_idx]
        row2 = df2.iloc[first_diff_idx]
        
        diff_count = 0
        for col in df1.columns:
            val1 = row1[col]
            val2 = row2[col]
            
            if pd.isna(val1) and pd.isna(val2):
                continue
            elif pd.isna(val1) or pd.isna(val2) or val1 != val2:
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if not np.isclose(val1, val2, rtol=1e-9, atol=1e-9):
                        if diff_count < 5:  # Show first 5 differences
                            print(f"  Column '{col}': {val1} vs {val2}")
                            diff_count += 1
                else:
                    if diff_count < 5:
                        print(f"  Column '{col}': {val1} vs {val2}")
                        diff_count += 1
        
        if diff_count >= 5:
            print(f"  ... (more differences in this row)")

def main():
    base_dir = "/localdisk2/pliu/park_multitask_fusion-main/data"
    
    # File pairs to compare
    comparisons = [
        {
            'name': 'Finger Tapping Features',
            'file1': f"{base_dir}/aligned_synthetic_format/mixed_scenario_4_real_finger_quick/features_demography_diagnosis_Nov22_2023.csv",
            'file2': f"{base_dir}/finger_tapping/features_demography_diagnosis_Nov22_2023.csv"
        },
        {
            'name': 'Audio (WavLM) Features',
            'file1': f"{base_dir}/aligned_synthetic_format/mixed_scenario_4_real_finger_quick/wavlm_fox_features.csv",
            'file2': f"{base_dir}/quick_brown_fox/wavlm_fox_features.csv"
        },
        {
            'name': 'Facial Expression Features',
            'file1': f"{base_dir}/aligned_synthetic_format/mixed_scenario_4_real_finger_quick/facial_dataset.csv",
            'file2': f"{base_dir}/facial_expression_smile/facial_dataset.csv"
        }
    ]
    
    print("="*70)
    print("COMPARING MIXED_SCENARIO_4 FILES WITH ORIGINAL FILES")
    print("="*70)
    
    for comp in comparisons:
        compare_csv_files(comp['file1'], comp['file2'], comp['name'])
    
    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()



