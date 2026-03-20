#!/usr/bin/env python
"""
Run VAE and Diffusion synthetic data validation experiments

6 experiments total:
- VAE: 3 experiments (2 real + 1 VAE synthetic)
- Diffusion: 3 experiments (2 real + 1 Diffusion synthetic)

Each method tests:
1. Real Smile + Real Finger + Synthetic Speech
2. Real Smile + Synthetic Finger + Real Speech  
3. Synthetic Smile + Real Finger + Real Speech
"""

import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent.parent.absolute()
RESULTS_DIR = BASE_DIR / "results" / "synthetic_validation" / "vae_diffusion_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Experiment configurations
EXPERIMENTS = {
    # VAE experiments
    'vae_exp4_smile_finger_synth_speech': {
        'method': 'vae',
        'description': 'VAE: Real Smile + Real Finger + Synthetic Speech',
        'finger': 'real',
        'smile': 'real', 
        'speech': 'vae_synthetic',
        'modalities': ['smile', 'finger', 'speech']
    },
    'vae_exp5_smile_synth_finger_speech': {
        'method': 'vae',
        'description': 'VAE: Real Smile + Synthetic Finger + Real Speech',
        'finger': 'vae_synthetic',
        'smile': 'real',
        'speech': 'real',
        'modalities': ['smile', 'finger', 'speech']
    },
    'vae_exp6_synth_smile_finger_speech': {
        'method': 'vae',
        'description': 'VAE: Synthetic Smile + Real Finger + Real Speech',
        'finger': 'real',
        'smile': 'vae_synthetic',
        'speech': 'real',
        'modalities': ['smile', 'finger', 'speech']
    },
    
    # Diffusion experiments
    'diff_exp4_smile_finger_synth_speech': {
        'method': 'diffusion',
        'description': 'Diffusion: Real Smile + Real Finger + Synthetic Speech',
        'finger': 'real',
        'smile': 'real',
        'speech': 'diffusion_synthetic',
        'modalities': ['smile', 'finger', 'speech']
    },
    'diff_exp5_smile_synth_finger_speech': {
        'method': 'diffusion',
        'description': 'Diffusion: Real Smile + Synthetic Finger + Real Speech',
        'finger': 'diffusion_synthetic',
        'smile': 'real',
        'speech': 'real',
        'modalities': ['smile', 'finger', 'speech']
    },
    'diff_exp6_synth_smile_finger_speech': {
        'method': 'diffusion',
        'description': 'Diffusion: Synthetic Smile + Real Finger + Real Speech',
        'finger': 'real',
        'smile': 'diffusion_synthetic',
        'speech': 'real',
        'modalities': ['smile', 'finger', 'speech']
    }
}

# File paths mapping
DATA_PATHS = {
    'real': {
        'finger': 'data/finger_tapping/features_demography_diagnosis_Nov22_2023.csv',
        'smile': 'data/facial_expression_smile/facial_dataset.csv',
        'speech': 'data/quick_brown_fox/wavlm_fox_features.csv'
    },
    'vae_synthetic': {
        'finger': 'data/synthetic_data/vae_synthetic/features_demography_diagnosis.csv',
        'smile': 'data/synthetic_data/vae_synthetic/facial_dataset.csv',
        'speech': 'data/synthetic_data/vae_synthetic/wavlm_fox_features.csv'
    },
    'diffusion_synthetic': {
        'finger': 'data/synthetic_data/diffusion_synthetic/features_demography_diagnosis.csv',
        'smile': 'data/synthetic_data/diffusion_synthetic/facial_dataset.csv',
        'speech': 'data/synthetic_data/diffusion_synthetic/wavlm_fox_features.csv'
    }
}


def generate_constants_file(exp_name, exp_config):
    """Generate constants.py file for an experiment"""
    constants_content = f"""import os

BASE_DIR = os.getcwd()+"/../../../"

# {exp_config['description']}
FINGER_FEATURES_FILE = os.path.join(BASE_DIR, "{DATA_PATHS[exp_config['finger']]['finger']}")
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "{DATA_PATHS[exp_config['smile']]['smile']}")
AUDIO_FEATURES_FILE = os.path.join(BASE_DIR, "{DATA_PATHS[exp_config['speech']]['speech']}")

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
    print(f"{'='*80}\n")
    
    # Generate constants file
    constants_content = generate_constants_file(exp_name, exp_config)
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
    cmd = ["conda", "run", "-n", "park", "python", "uncertainty_aware_fusion.py"]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")
    
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
        
        # Extract results
        if log_file.exists():
            with open(log_file, 'r') as f:
                log_content = f.read()
            if "'auroc'" in log_content:
                # Try to extract AUROC
                lines = log_content.split('\n')
                for line in lines[-50:]:
                    if "'auroc'" in line.lower() or 'auroc' in line.lower():
                        print(f"Found: {line.strip()}")
        
        print(f"✅ Experiment {exp_name} completed (exit code: {result.returncode})")
        return result.returncode == 0
        
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
    print(f"# Total experiments: {len(experiments_to_run)}")
    print(f"{'#'*80}\n")
    
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

