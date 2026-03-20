#!/usr/bin/env python
"""
Convert new synthetic data format to fusion model expected format

New format: Only feature columns + Participant_ID, date, pd, hand
Expected format: Feature columns + Filename, Protocol, Participant_ID, Task, etc.
"""

import pandas as pd
import os
from pathlib import Path

def convert_finger_data(input_file, output_file):
    """Convert finger data to expected format"""
    df = pd.read_csv(input_file)
    
    # Handle hand='both' case: split into left and right rows
    if 'hand' in df.columns and (df['hand'] == 'both').any():
        # Duplicate rows with hand='both' to create left and right versions
        df_both = df[df['hand'] == 'both'].copy()
        df_left = df_both.copy()
        df_left['hand'] = 'left'
        df_right = df_both.copy()
        df_right['hand'] = 'right'
        # Remove 'both' rows and add left/right versions
        df = df[df['hand'] != 'both']
        df = pd.concat([df, df_left, df_right], ignore_index=True)
    
    # Create filename column (lowercase!) from Participant_ID and date
    # Format: {date}_{Participant_ID}_finger_tapping_{hand}.mp4
    df['filename'] = df.apply(
        lambda row: f"{row['date']}_{row['Participant_ID']}_finger_tapping_{row['hand']}.mp4",
        axis=1
    )
    
    # Add required columns with default values
    df['Protocol'] = 'SuperPD'
    df['Task'] = df.apply(lambda row: f"finger_tapping_{row['hand']}", axis=1)
    df['Duration'] = 0
    df['FPS'] = 30.0
    df['Frame_Height'] = 480.0
    df['Frame_Width'] = 640.0
    df['gender'] = ''
    df['age'] = ''
    df['race'] = ''
    df['ethnicity'] = ''
    df['dob'] = ''
    df['time_mdsupdrs'] = ''
    
    # Reorder columns: metadata first, then features
    feature_cols = [c for c in df.columns if c.startswith('finger_feature_')]
    metadata_cols = ['Unnamed: 0', 'filename', 'Protocol', 'Participant_ID', 'Task', 
                     'Duration', 'FPS', 'Frame_Height', 'Frame_Width', 'gender', 'age',
                     'race', 'ethnicity', 'pd', 'dob', 'time_mdsupdrs', 'date', 'hand']
    
    # Add index as Unnamed: 0
    df.insert(0, 'Unnamed: 0', range(len(df)))
    
    # Reorder: metadata + features
    all_cols = [c for c in metadata_cols if c in df.columns] + sorted(feature_cols)
    df = df[all_cols]
    
    df.to_csv(output_file, index=False)
    print(f"✅ Converted finger data: {len(df)} rows")
    return df


def convert_smile_data(input_file, output_file):
    """Convert smile data to expected format"""
    df = pd.read_csv(input_file)
    
    # Check if ID and date columns exist
    if 'ID' not in df.columns and 'Participant_ID' in df.columns:
        df['ID'] = df['Participant_ID']
    if 'date' not in df.columns:
        df['date'] = '2024-01-01'  # Default date
    
    # Create Filename column (uppercase F - required by fusion script!)
    df['Filename'] = df.apply(
        lambda row: f"{row['date']}_{row['ID']}_smile.mp4",
        axis=1
    )
    
    # Add required metadata columns
    # Fusion script expects 'ID' column for smile data
    if 'ID' not in df.columns:
        df['ID'] = df.get('Participant_ID', df.get('ID', range(len(df))))
    if 'Participant_ID' not in df.columns:
        df['Participant_ID'] = df['ID']
    df['gender'] = ''
    df['age'] = ''
    df['race'] = ''
    
    # Reorder columns: Filename, ID, Participant_ID, gender, age, race first, then features, then pd
    feature_cols = [c for c in df.columns if c.startswith('smile_feature_')]
    df = df[['Filename', 'ID', 'Participant_ID', 'gender', 'age', 'race'] + feature_cols + ['pd']]
    
    df.to_csv(output_file, index=False)
    print(f"✅ Converted smile data: {len(df)} rows")
    return df


def convert_speech_data(input_file, output_file):
    """Convert speech data to expected format"""
    df = pd.read_csv(input_file)
    
    # Check if Participant_ID and date columns exist
    if 'Participant_ID' not in df.columns:
        df['Participant_ID'] = df.get('ID', range(len(df)))
    if 'date' not in df.columns:
        df['date'] = '2024-01-01'  # Default date
    
    # Create Filename column (uppercase F - required by fusion script!)
    df['Filename'] = df.apply(
        lambda row: f"{row['date']}_{row['Participant_ID']}_quick_brown_fox.mp4",
        axis=1
    )
    
    # Add required metadata columns
    df['gender'] = ''
    df['age'] = ''
    df['race'] = ''
    
    # Reorder columns: Filename, Participant_ID, gender, age, race first, then features, then pd
    feature_cols = [c for c in df.columns if c.startswith('wavlm_feature')]
    df = df[['Filename', 'Participant_ID', 'gender', 'age', 'race'] + sorted(feature_cols) + ['pd']]
    
    df.to_csv(output_file, index=False)
    print(f"✅ Converted speech data: {len(df)} rows")
    return df


def convert_experiment_folder(exp_folder, output_base_dir):
    """Convert all files in an experiment folder"""
    exp_folder = Path(exp_folder)
    output_dir = Path(output_base_dir) / exp_folder.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Converting: {exp_folder.name}")
    print(f"{'='*60}")
    
    # Convert each file
    finger_file = exp_folder / "features_demography_diagnosis.csv"
    smile_file = exp_folder / "facial_dataset.csv"
    speech_file = exp_folder / "wavlm_fox_features.csv"
    
    if finger_file.exists():
        convert_finger_data(finger_file, output_dir / "features_demography_diagnosis.csv")
    if smile_file.exists():
        convert_smile_data(smile_file, output_dir / "facial_dataset.csv")
    if speech_file.exists():
        convert_speech_data(speech_file, output_dir / "wavlm_fox_features.csv")
    
    print(f"✅ Converted to: {output_dir}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert new synthetic data format')
    parser.add_argument('--input', type=str, required=True,
                       help='Input experiment folder')
    parser.add_argument('--output', type=str, required=True,
                       help='Output base directory')
    
    args = parser.parse_args()
    
    convert_experiment_folder(args.input, args.output)

