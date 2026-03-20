#!/usr/bin/env python
"""
Detailed comparison showing which specific rows differ in facial dataset
"""
import pandas as pd
import numpy as np

def detailed_facial_comparison():
    base_dir = "/localdisk2/pliu/park_multitask_fusion-main/data"
    
    file1 = f"{base_dir}/aligned_synthetic_format/mixed_scenario_4_real_finger_quick/facial_dataset.csv"
    file2 = f"{base_dir}/facial_expression_smile/facial_dataset.csv"
    
    print("="*70)
    print("DETAILED FACIAL DATASET COMPARISON")
    print("="*70)
    
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    print(f"\nFile 1 (Scenario 4): {len(df1)} rows")
    print(f"File 2 (Original):   {len(df2)} rows")
    
    # Find different rows with their identifiers
    different_rows_info = []
    
    for idx in range(len(df1)):
        row1 = df1.iloc[idx]
        row2 = df2.iloc[idx]
        
        differs = False
        for col in df1.columns:
            val1 = row1[col]
            val2 = row2[col]
            
            if pd.isna(val1) and pd.isna(val2):
                continue
            elif pd.isna(val1) or pd.isna(val2):
                differs = True
                break
            elif val1 != val2:
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if not np.isclose(val1, val2, rtol=1e-9, atol=1e-9):
                        differs = True
                        break
                else:
                    differs = True
                    break
        
        if differs:
            # Get identifier columns (usually Filename or similar)
            identifier = row1.get('Filename', row1.get('id', idx))
            different_rows_info.append({
                'row_index': idx,
                'identifier': identifier
            })
    
    print(f"\nTotal different rows: {len(different_rows_info)}")
    print(f"Percentage: {len(different_rows_info)/len(df1)*100:.2f}%")
    
    if len(different_rows_info) > 0:
        print(f"\nFirst 20 different rows:")
        print(f"{'Row Index':<12} {'Identifier':<50}")
        print("-" * 70)
        for info in different_rows_info[:20]:
            print(f"{info['row_index']:<12} {str(info['identifier']):<50}")
        
        if len(different_rows_info) > 20:
            print(f"... and {len(different_rows_info) - 20} more")
        
        # Check if these are synthetic data
        print(f"\n{'='*70}")
        print("ANALYSIS: Synthetic vs Real Data")
        print(f"{'='*70}")
        print(f"\nThe {len(different_rows_info)} different rows ({len(different_rows_info)/len(df1)*100:.2f}%) are likely:")
        print(f"  ✓ SYNTHETIC data (generated/augmented) in Scenario 4")
        print(f"  ✓ REAL data in the original dataset")
        print(f"\nThis aligns with the scenario design where scenario 4 uses:")
        print(f"  - Real Finger Tapping features")
        print(f"  - Real Quick audio features")
        print(f"  - Synthetic Smile (facial) features")

def main():
    detailed_facial_comparison()

if __name__ == "__main__":
    main()



