#!/usr/bin/env python3
"""
Compare facial_dataset.csv between scenario 10 and original
Based on filename (unique identifier) to match rows
"""

import pandas as pd
import numpy as np

def compare_facial_datasets_by_filename():
    """Compare two facial dataset CSV files based on filename"""
    
    # File paths
    scenario10_file = "/localdisk2/pliu/park_multitask_fusion-main/data/aligned_synthetic_format/add_synthetic_to_train_data_with_new_rows/facial_dataset.csv"
    original_file = "/localdisk2/pliu/park_multitask_fusion-main/data/facial_expression_smile/facial_dataset.csv"
    
    print("=" * 80)
    print("Comparing Facial Dataset Files (By Filename)")
    print("=" * 80)
    print(f"\nScenario 10: {scenario10_file}")
    print(f"Original:    {original_file}")
    print()
    
    # Read both files
    try:
        df_scenario10 = pd.read_csv(scenario10_file)
        df_original = pd.read_csv(original_file)
    except Exception as e:
        print(f"❌ Error reading files: {e}")
        return
    
    print(f"Scenario 10 rows: {len(df_scenario10)}")
    print(f"Original rows:    {len(df_original)}")
    print(f"Difference:       {len(df_scenario10) - len(df_original)} rows")
    print()
    
    # Check if 'Filename' column exists (capital F)
    filename_col = None
    for col in ['Filename', 'filename', 'file_name']:
        if col in df_scenario10.columns and col in df_original.columns:
            filename_col = col
            break
    
    if filename_col is None:
        print("❌ Could not find matching filename column")
        print(f"Scenario 10 columns: {list(df_scenario10.columns[:10])}...")
        print(f"Original columns: {list(df_original.columns[:10])}...")
        return
    
    print(f"✅ Both files have '{filename_col}' column")
    print(f"Columns: {len(df_scenario10.columns)}")
    print()
    
    # Create sets of filenames
    filenames_scenario10 = set(df_scenario10[filename_col].values)
    filenames_original = set(df_original[filename_col].values)
    
    # Find common and unique filenames
    common_filenames = filenames_scenario10 & filenames_original
    only_in_scenario10 = filenames_scenario10 - filenames_original
    only_in_original = filenames_original - filenames_scenario10
    
    print("-" * 80)
    print("Filename Analysis")
    print("-" * 80)
    print(f"Common filenames (in both):        {len(common_filenames)}")
    print(f"Only in Scenario 10 (new):         {len(only_in_scenario10)}")
    print(f"Only in Original (missing):        {len(only_in_original)}")
    print()
    
    # For common filenames, compare row values
    print("-" * 80)
    print("Row Content Comparison (for common filenames)")
    print("-" * 80)
    
    # Set Filename as index for easier lookup
    df_scenario10_indexed = df_scenario10.set_index(filename_col)
    df_original_indexed = df_original.set_index(filename_col)
    
    identical_rows = 0
    different_rows = 0
    different_examples = []
    
    for filename in sorted(common_filenames):
        row_scenario10 = df_scenario10_indexed.loc[filename]
        row_original = df_original_indexed.loc[filename]
        
        # Compare all column values
        if row_scenario10.equals(row_original):
            identical_rows += 1
        else:
            different_rows += 1
            if len(different_examples) < 5:  # Keep first 5 examples
                # Find which columns differ
                diff_cols = []
                for col in row_scenario10.index:
                    val1 = row_scenario10[col]
                    val2 = row_original[col]
                    # Handle NaN comparison
                    if pd.isna(val1) and pd.isna(val2):
                        continue
                    if val1 != val2:
                        diff_cols.append((col, val1, val2))
                
                different_examples.append({
                    'filename': filename,
                    'diff_cols': diff_cols
                })
    
    print(f"\n📊 Comparison Results (for {len(common_filenames)} common files):")
    if len(common_filenames) > 0:
        print(f"  ✅ Identical rows:  {identical_rows} ({100 * identical_rows / len(common_filenames):.2f}%)")
        print(f"  ❌ Different rows:  {different_rows} ({100 * different_rows / len(common_filenames):.2f}%)")
    else:
        print(f"  No common filenames to compare!")
    print()
    
    # Show examples of different rows
    if different_rows > 0:
        print("=" * 80)
        print(f"Examples of Different Rows (showing up to 5)")
        print("=" * 80)
        for i, example in enumerate(different_examples, 1):
            print(f"\n{i}. Filename: {example['filename']}")
            print(f"   Columns that differ ({len(example['diff_cols'])}):")
            for col, val1, val2 in example['diff_cols'][:10]:  # Show max 10 columns
                print(f"     - {col}:")
                print(f"         Scenario 10: {val1}")
                print(f"         Original:    {val2}")
            if len(example['diff_cols']) > 10:
                print(f"     ... and {len(example['diff_cols']) - 10} more columns")
    
    # Show rows only in Scenario 10 (new synthetic data)
    if len(only_in_scenario10) > 0:
        print("\n" + "=" * 80)
        print(f"Rows ONLY in Scenario 10 (New Synthetic Data): {len(only_in_scenario10)}")
        print("=" * 80)
        
        # Get full rows for these filenames
        new_rows = df_scenario10[df_scenario10[filename_col].isin(only_in_scenario10)]
        
        print(f"\nShowing first {min(10, len(new_rows))} new rows:")
        for idx, (_, row) in enumerate(new_rows.head(10).iterrows(), 1):
            print(f"\n{idx}. Filename: {row[filename_col]}")
            # Show key columns
            key_cols = ['Participant_ID', 'date', 'Protocol', 'Task', 'Diagnosis']
            for col in key_cols:
                if col in row.index:
                    print(f"   {col}: {row[col]}")
    
    # Show rows only in Original (missing from Scenario 10)
    if len(only_in_original) > 0:
        print("\n" + "=" * 80)
        print(f"⚠️  Rows ONLY in Original (Missing from Scenario 10): {len(only_in_original)}")
        print("=" * 80)
        
        missing_rows = df_original[df_original[filename_col].isin(only_in_original)]
        print(f"\nShowing first {min(5, len(missing_rows))} missing rows:")
        for idx, (_, row) in enumerate(missing_rows.head(5).iterrows(), 1):
            print(f"\n{idx}. Filename: {row[filename_col]}")
            key_cols = ['Participant_ID', 'date', 'Protocol', 'Task', 'Diagnosis']
            for col in key_cols:
                if col in row.index:
                    print(f"   {col}: {row[col]}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n📊 Data Composition:")
    print(f"  Original file:           {len(df_original)} rows")
    print(f"  Scenario 10 file:        {len(df_scenario10)} rows")
    print(f"  Net change:              {len(df_scenario10) - len(df_original):+d} rows")
    print()
    print(f"📊 Filename Matching:")
    print(f"  Common (in both):        {len(common_filenames)} filenames")
    print(f"  New (only in Sc10):      {len(only_in_scenario10)} filenames (synthetic)")
    print(f"  Missing (only in Orig):  {len(only_in_original)} filenames")
    print()
    print(f"📊 Content Comparison (for common filenames):")
    if len(common_filenames) > 0:
        print(f"  Identical rows:          {identical_rows}/{len(common_filenames)} ({100 * identical_rows / len(common_filenames):.2f}%)")
        print(f"  Modified rows:           {different_rows}/{len(common_filenames)} ({100 * different_rows / len(common_filenames):.2f}%)")
    else:
        print(f"  No common filenames to compare")
    print()
    
    if len(only_in_scenario10) > 0 and len(only_in_original) == 0 and different_rows == 0:
        print("✅ CONCLUSION: Scenario 10 = Original + New Synthetic Rows")
        print(f"   • All original {len(common_filenames)} rows are preserved (100% identical)")
        print(f"   • {len(only_in_scenario10)} new synthetic rows added")
        print(f"   • No data replaced or modified")
    elif different_rows > 0:
        print("⚠️  CONCLUSION: Some original rows have been modified")
        print(f"   • {identical_rows} rows unchanged")
        print(f"   • {different_rows} rows modified")
        print(f"   • {len(only_in_scenario10)} new rows added")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    compare_facial_datasets_by_filename()

