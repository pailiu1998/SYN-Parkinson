#!/usr/bin/env python
"""
Check if all required data files are ready for experiments
"""
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent.parent.absolute()

REQUIRED_FILES = {
    'vae': {
        'finger': BASE_DIR / 'data/synthetic_data/vae_synthetic/features_demography_diagnosis.csv',
        'smile': BASE_DIR / 'data/synthetic_data/vae_synthetic/facial_dataset.csv',
        'speech': BASE_DIR / 'data/synthetic_data/vae_synthetic/wavlm_fox_features.csv'
    },
    'diffusion': {
        'finger': BASE_DIR / 'data/synthetic_data/diffusion_synthetic/features_demography_diagnosis.csv',
        'smile': BASE_DIR / 'data/synthetic_data/diffusion_synthetic/facial_dataset.csv',
        'speech': BASE_DIR / 'data/synthetic_data/diffusion_synthetic/wavlm_fox_features.csv'
    },
    'real': {
        'finger': BASE_DIR / 'data/finger_tapping/features_demography_diagnosis_Nov22_2023.csv',
        'smile': BASE_DIR / 'data/facial_expression_smile/facial_dataset.csv',
        'speech': BASE_DIR / 'data/quick_brown_fox/wavlm_fox_features.csv'
    }
}

def check_files():
    """Check if all required files exist"""
    print("="*80)
    print("DATA READINESS CHECK")
    print("="*80)
    
    all_ready = True
    
    for data_type, files in REQUIRED_FILES.items():
        print(f"\n{data_type.upper()} Data:")
        print("-" * 40)
        for modality, filepath in files.items():
            exists = filepath.exists()
            size = filepath.stat().st_size / (1024*1024) if exists else 0
            status = "✅" if exists else "❌"
            print(f"  {status} {modality:10s}: {filepath.name}")
            if exists:
                print(f"      Size: {size:.2f} MB")
            else:
                print(f"      Missing!")
                all_ready = False
    
    print("\n" + "="*80)
    if all_ready:
        print("✅ All data files are ready! You can run experiments now.")
    else:
        print("❌ Some data files are missing. Please prepare them first.")
    print("="*80)
    
    return all_ready

if __name__ == '__main__':
    check_files()


