#!/usr/bin/env python3
"""
Compare facial_dataset.csv between scenario 10 (add synthetic) and original
"""

import pandas as pd
import sys

def compare_facial_datasets():
    """Compare two facial dataset CSV files line by line"""
    
    # File paths
    scenario10_file = "/localdisk2/pliu/park_multitask_fusion-main/data/aligned_synthetic_format/add_synthetic_to_train_data_with_new_rows/facial_dataset.csv"
    original_file = "/localdisk2/pliu/park_multitask_fusion-main/data/facial_expression_smile/facial_dataset.csv"
    
    print("=" * 80)
    print("Comparing Facial Dataset Files")
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
    
    # Check columns
    if list(df_scenario10.columns) != list(df_original.columns):
        print("⚠️  WARNING: Column names differ!")
        print(f"Scenario 10 columns: {list(df_scenario10.columns)}")
        print(f"Original columns:    {list(df_original.columns)}")
        print()
    else:
        print(f"✅ Column names match ({len(df_scenario10.columns)} columns)")
        print()
    
    # Strategy: Compare based on row content
    # For rows that exist in both files (up to min length)
    min_rows = min(len(df_scenario10), len(df_original))
    max_rows = max(len(df_scenario10), len(df_original))
    
    print("-" * 80)
    print("Row-by-Row Comparison")
    print("-" * 80)
    
    # Compare common rows
    different_rows = 0
    identical_rows = 0
    
    for i in range(min_rows):
        row1 = df_scenario10.iloc[i]
        row2 = df_original.iloc[i]
        
        if not row1.equals(row2):
            different_rows += 1
        else:
            identical_rows += 1
    
    # Additional rows (only in longer file)
    additional_rows = max_rows - min_rows
    
    print(f"\n📊 Comparison Results:")
    print(f"  Identical rows (same position, same content): {identical_rows}")
    print(f"  Different rows (same position, diff content): {different_rows}")
    print(f"  Additional rows (only in Scenario 10):        {additional_rows}")
    print()
    
    print(f"✅ Total rows in common range:     {min_rows}")
    print(f"📈 Identical percentage:            {100 * identical_rows / min_rows:.2f}%")
    print(f"📉 Different percentage:            {100 * different_rows / min_rows:.2f}%")
    print()
    
    if additional_rows > 0:
        print(f"⭐ Scenario 10 has {additional_rows} EXTRA rows (new synthetic data added)")
        print(f"   This confirms the 'add synthetic' approach (vs. 'replace' in other scenarios)")
    elif additional_rows < 0:
        print(f"⚠️  Original has {abs(additional_rows)} more rows than Scenario 10")
    
    print()
    print("=" * 80)
    print("Analysis Summary")
    print("=" * 80)
    
    total_scenario10 = len(df_scenario10)
    
    if additional_rows > 0:
        print(f"\n✅ Scenario 10 = Original ({len(df_original)}) + Synthetic ({additional_rows}) = {total_scenario10}")
        print(f"\n📈 Data Augmentation:")
        print(f"   - Base real data:     {len(df_original)} rows (100%)")
        print(f"   - Added synthetic:    {additional_rows} rows ({100 * additional_rows / len(df_original):.2f}% increase)")
        print(f"   - Total training:     {total_scenario10} rows")
        
        if different_rows > 0:
            print(f"\n⚠️  Note: {different_rows} rows differ between files (may be due to data preprocessing)")
    else:
        print(f"\n⚠️  Unexpected: Scenario 10 doesn't have more rows than Original")
        print(f"   Expected behavior: add_synthetic_to_train_data should ADD rows, not replace")
    
    print("\n" + "=" * 80)
    
    # Show a few examples of differences
    if different_rows > 0 and different_rows <= 10:
        print("\n📋 Examples of different rows:")
        print("-" * 80)
        diff_count = 0
        for i in range(min_rows):
            if diff_count >= 5:  # Show max 5 examples
                break
            row1 = df_scenario10.iloc[i]
            row2 = df_original.iloc[i]
            if not row1.equals(row2):
                print(f"\nRow {i}:")
                print(f"  Scenario 10: {row1.values[:5]}...")
                print(f"  Original:    {row2.values[:5]}...")
                diff_count += 1
    
    # Show some additional rows if they exist
    if additional_rows > 0:
        print("\n📋 Examples of ADDITIONAL rows in Scenario 10 (new synthetic data):")
        print("-" * 80)
        show_count = min(5, additional_rows)
        for i in range(show_count):
            row_idx = len(df_original) + i
            row = df_scenario10.iloc[row_idx]
            print(f"\nAdditional Row {row_idx}:")
            print(f"  {row.values[:8]}...")

if __name__ == "__main__":
    compare_facial_datasets()

