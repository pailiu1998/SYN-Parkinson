#!/usr/bin/env python
"""
Compare files between mixed_scenario_6 and original data folders
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
    
    print(f"File 1 (Scenario 6): {file1.split('/')[-1]}")
    print(f"  Rows: {len(df1)}, Columns: {len(df1.columns)}")
    print(f"File 2 (Original):   {file2.split('/')[-1]}")
    print(f"  Rows: {len(df2)}, Columns: {len(df2.columns)}")
    
    # Check if shapes match
    if df1.shape != df2.shape:
        print(f"⚠ WARNING: Shapes don't match!")
        return
    
    # Check if column names match
    if not all(df1.columns == df2.columns):
        print(f"⚠ WARNING: Column names don't match!")
        return
    
    # Compare row by row
    different_rows = 0
    same_rows = 0
    different_row_indices = []
    
    for idx in range(len(df1)):
        row1 = df1.iloc[idx]
        row2 = df2.iloc[idx]
        
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
            if len(different_row_indices) < 10:
                different_row_indices.append(idx)
        else:
            same_rows += 1
    
    total_rows = len(df1)
    percentage_different = (different_rows / total_rows) * 100 if total_rows > 0 else 0
    percentage_same = (same_rows / total_rows) * 100 if total_rows > 0 else 0
    
    print(f"\nResults:")
    print(f"  Total rows: {total_rows}")
    print(f"  Identical rows: {same_rows} ({percentage_same:.2f}%)")
    print(f"  Different rows: {different_rows} ({percentage_different:.2f}%)")
    
    if different_rows == 0:
        print(f"  ✅ Status: 100% IDENTICAL - REAL DATA")
    else:
        print(f"  ⚠️  Status: MIXED DATA ({percentage_same:.2f}% real, {percentage_different:.2f}% synthetic)")
    
    if different_rows > 0 and len(different_row_indices) > 0:
        print(f"\nFirst few different row indices: {different_row_indices}")
        
        # Show details of first different row
        first_diff_idx = different_row_indices[0]
        print(f"\nExample - Row {first_diff_idx} differences:")
        row1 = df1.iloc[first_diff_idx]
        row2 = df2.iloc[first_diff_idx]
        
        diff_count = 0
        for col in df1.columns[:10]:  # Show first 10 columns
            val1 = row1[col]
            val2 = row2[col]
            
            if pd.isna(val1) and pd.isna(val2):
                continue
            elif pd.isna(val1) or pd.isna(val2) or val1 != val2:
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if not np.isclose(val1, val2, rtol=1e-9, atol=1e-9):
                        if diff_count < 5:
                            print(f"  Column '{col}': {val1} vs {val2}")
                            diff_count += 1
                else:
                    if diff_count < 5:
                        print(f"  Column '{col}': {val1} vs {val2}")
                        diff_count += 1
    
    return {
        'total': total_rows,
        'same': same_rows,
        'different': different_rows,
        'percentage_different': percentage_different
    }

def main():
    base_dir = "/localdisk2/pliu/park_multitask_fusion-main/data"
    
    print("="*70)
    print("COMPARING MIXED_SCENARIO_6 FILES WITH ORIGINAL FILES")
    print("="*70)
    print("\nScenario 6: real_quick_smile")
    print("Expected: Real Quick + Real Smile, Synthetic Finger")
    print("="*70)
    
    # File pairs to compare
    comparisons = [
        {
            'name': 'Finger Tapping Features',
            'file1': f"{base_dir}/aligned_synthetic_format/mixed_scenario_6_real_quick_smile/features_demography_diagnosis.csv",
            'file2': f"{base_dir}/finger_tapping/features_demography_diagnosis_Nov22_2023.csv"
        },
        {
            'name': 'Audio (WavLM) Features',
            'file1': f"{base_dir}/aligned_synthetic_format/mixed_scenario_6_real_quick_smile/wavlm_fox_features.csv",
            'file2': f"{base_dir}/quick_brown_fox/wavlm_fox_features.csv"
        },
        {
            'name': 'Facial Expression Features',
            'file1': f"{base_dir}/aligned_synthetic_format/mixed_scenario_6_real_quick_smile/facial_dataset.csv",
            'file2': f"{base_dir}/facial_expression_smile/facial_dataset.csv"
        }
    ]
    
    results = {}
    for comp in comparisons:
        result = compare_csv_files(comp['file1'], comp['file2'], comp['name'])
        results[comp['name']] = result
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY - SCENARIO 6 CONFIGURATION")
    print(f"{'='*70}")
    print(f"\n{'Modality':<25} {'Status':<20} {'Real %':<10} {'Synthetic %':<15}")
    print("-" * 70)
    
    for name, result in results.items():
        if result:
            modality = name.split(' ')[0]
            if result['different'] == 0:
                status = "✅ 100% Real"
                real_pct = "100%"
                synth_pct = "0%"
            else:
                status = "⚠️  Mixed"
                real_pct = f"{100-result['percentage_different']:.2f}%"
                synth_pct = f"{result['percentage_different']:.2f}%"
            
            print(f"{modality:<25} {status:<20} {real_pct:<10} {synth_pct:<15}")
    
    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()



