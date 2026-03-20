#!/usr/bin/env python
"""
Run VAE and Diffusion synthetic data validation experiments

6 experiments total (2real + 1synth):
- VAE: 3 experiments
- Diffusion: 3 experiments

Data structure:
fusion_test_csv_format/
├── vae/2real_1synth/
│   ├── 1_smile_quick_REAL__finger_SYNTH/
│   ├── 2_smile_finger_REAL__quick_SYNTH/
│   └── 3_quick_finger_REAL__smile_SYNTH/
└── diffusion/2real_1synth/
    ├── 1_smile_quick_REAL__finger_SYNTH/
    ├── 2_smile_finger_REAL__quick_SYNTH/
    └── 3_quick_finger_REAL__smile_SYNTH/
"""

import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent.parent.absolute()
CONVERTED_DATA_DIR = BASE_DIR / "data" / "synthetic_data" / "converted"
RESULTS_DIR = BASE_DIR / "results" / "synthetic_validation" / "vae_diffusion_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Experiment configurations matching the folder structure
EXPERIMENTS = {
    # VAE experiments
    'vae_exp1_smile_quick_real_finger_synth': {
        'method': 'vae',
        'folder': 'vae_2real_1synth/1_smile_quick_REAL__finger_SYNTH',
        'description': 'VAE: Real Smile + Real Quick + Synthetic Finger',
        'modalities': ['smile', 'finger', 'speech']
    },
    'vae_exp2_smile_finger_real_quick_synth': {
        'method': 'vae',
        'folder': 'vae_2real_1synth/2_smile_finger_REAL__quick_SYNTH',
        'description': 'VAE: Real Smile + Real Finger + Synthetic Quick',
        'modalities': ['smile', 'finger', 'speech']
    },
    'vae_exp3_quick_finger_real_smile_synth': {
        'method': 'vae',
        'folder': 'vae_2real_1synth/3_quick_finger_REAL__smile_SYNTH',
        'description': 'VAE: Real Quick + Real Finger + Synthetic Smile',
        'modalities': ['smile', 'finger', 'speech']
    },
    
    # Diffusion experiments
    'diff_exp1_smile_quick_real_finger_synth': {
        'method': 'diffusion',
        'folder': 'diffusion_2real_1synth/1_smile_quick_REAL__finger_SYNTH',
        'description': 'Diffusion: Real Smile + Real Quick + Synthetic Finger',
        'modalities': ['smile', 'finger', 'speech']
    },
    'diff_exp2_smile_finger_real_quick_synth': {
        'method': 'diffusion',
        'folder': 'diffusion_2real_1synth/2_smile_finger_REAL__quick_SYNTH',
        'description': 'Diffusion: Real Smile + Real Finger + Synthetic Quick',
        'modalities': ['smile', 'finger', 'speech']
    },
    'diff_exp3_quick_finger_real_smile_synth': {
        'method': 'diffusion',
        'folder': 'diffusion_2real_1synth/3_quick_finger_REAL__smile_SYNTH',
        'description': 'Diffusion: Real Quick + Real Finger + Synthetic Smile',
        'modalities': ['smile', 'finger', 'speech']
    }
}


def generate_constants_file(exp_name, exp_config):
    """Generate constants.py file for an experiment"""
    data_folder = CONVERTED_DATA_DIR / exp_config['folder']
    
    finger_file = data_folder / "features_demography_diagnosis.csv"
    smile_file = data_folder / "facial_dataset.csv"
    speech_file = data_folder / "wavlm_fox_features.csv"
    
    # Check if files exist
    missing = []
    if not finger_file.exists():
        missing.append(str(finger_file))
    if not smile_file.exists():
        missing.append(str(smile_file))
    if not speech_file.exists():
        missing.append(str(speech_file))
    
    if missing:
        raise FileNotFoundError(f"Missing files for {exp_name}:\n" + "\n".join(missing))
    
    # Use relative paths from BASE_DIR
    finger_rel = finger_file.relative_to(BASE_DIR)
    smile_rel = smile_file.relative_to(BASE_DIR)
    speech_rel = speech_file.relative_to(BASE_DIR)
    
    constants_content = f"""import os

BASE_DIR = os.getcwd()+"/../../../"

# {exp_config['description']}
FINGER_FEATURES_FILE = os.path.join(BASE_DIR, "{finger_rel}")
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "{smile_rel}")
AUDIO_FEATURES_FILE = os.path.join(BASE_DIR, "{speech_rel}")

MODEL_BASE_PATH = os.path.join(BASE_DIR,"models")

FACIAL_EXPRESSIONS = {{
    'smile': True,
    'surprise': False,
    'disgust': False
}}

MODEL_SUBSETS = {{
    0: ['finger_model_both_hand_fusion_baal', 'fox_model_best_auroc_baal', 'facial_expression_smile_best_auroc_baal'],
}}
"""
    return constants_content


def run_experiment(exp_name, exp_config):
    """Run a single experiment"""
    print(f"\n{'='*80}")
    print(f"Running: {exp_name}")
    print(f"Description: {exp_config['description']}")
    print(f"Data folder: {exp_config['folder']}")
    print(f"{'='*80}\n")
    
    # Generate constants file
    try:
        constants_content = generate_constants_file(exp_name, exp_config)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return False
    
    constants_path = Path(__file__).parent / "constants.py"
    
    # Backup original constants
    backup_path = Path(__file__).parent / "constants_backup_original.py"
    if not backup_path.exists():
        if constants_path.exists():
            with open(constants_path, 'r') as f:
                backup_content = f.read()
            with open(backup_path, 'w') as f:
                f.write(backup_content)
    
    # Write new constants
    with open(constants_path, 'w') as f:
        f.write(constants_content)
    
    # Run experiment
    log_file = RESULTS_DIR / f"{exp_name}.log"
    # Use python directly (assume environment is activated or use system python)
    cmd = ["python", "uncertainty_aware_fusion.py"]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")
    
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=3600  # 1 hour timeout
            )
        
        # Extract results
        if log_file.exists():
            with open(log_file, 'r') as f:
                log_content = f.read()
            if "'auroc'" in log_content or 'auroc' in log_content.lower():
                # Try to extract AUROC
                lines = log_content.split('\n')
                for line in lines[-50:]:
                    if "'auroc'" in line.lower() or ('auroc' in line.lower() and ':' in line):
                        print(f"Result: {line.strip()}")
        
        success = result.returncode == 0
        status = "✅" if success else "❌"
        print(f"{status} Experiment {exp_name} completed (exit code: {result.returncode})")
        return success
        
    except subprocess.TimeoutExpired:
        print(f"❌ Experiment {exp_name} timed out after 1 hour")
        return False
    except Exception as e:
        print(f"❌ Error running {exp_name}: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run VAE and Diffusion synthetic validation experiments')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all'] + list(EXPERIMENTS.keys()),
                       help='Which experiment to run')
    parser.add_argument('--method', type=str, choices=['vae', 'diffusion', 'both'],
                       help='Which method to run (vae, diffusion, or both)')
    
    args = parser.parse_args()
    
    # Filter experiments by method if specified
    experiments_to_run = {}
    if args.experiment == 'all':
        if args.method:
            experiments_to_run = {k: v for k, v in EXPERIMENTS.items() 
                                 if v['method'] == args.method or (args.method == 'both')}
        else:
            experiments_to_run = EXPERIMENTS
    else:
        experiments_to_run = {args.experiment: EXPERIMENTS[args.experiment]}
    
    print(f"\n{'#'*80}")
    print(f"# VAE vs Diffusion Synthetic Data Validation")
    print(f"# Converted data directory: {CONVERTED_DATA_DIR}")
    print(f"# Total experiments: {len(experiments_to_run)}")
    print(f"{'#'*80}\n")
    
    # Check data source exists
    if not CONVERTED_DATA_DIR.exists():
        print(f"❌ Converted data directory not found: {CONVERTED_DATA_DIR}")
        print("Please run convert_new_data_format.py first!")
        return
    
    results = {}
    for exp_name, exp_config in experiments_to_run.items():
        success = run_experiment(exp_name, exp_config)
        results[exp_name] = {
            'success': success,
            'config': exp_config
        }
    
    # Summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")
    for exp_name, result in results.items():
        status = "✅ Success" if result['success'] else "❌ Failed"
        print(f"{exp_name}: {status}")
    
    print(f"\n✅ All experiments completed!")
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == '__main__':
    main()

