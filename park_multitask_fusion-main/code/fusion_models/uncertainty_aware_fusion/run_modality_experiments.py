#!/usr/bin/env python
"""
Run different modality combination experiments
"""
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from uncertainty_aware_fusion import main

# Experiment configurations
EXPERIMENTS = {
    'exp1_smile_finger': {
        'model_subset_choice': 2,  # Smile + Finger
        'description': 'Two modality: Smile + Finger'
    },
    'exp2_smile_speech': {
        'model_subset_choice': 3,  # Smile + Speech  
        'description': 'Two modality: Smile + Speech'
    },
    'exp3_finger_speech': {
        'model_subset_choice': 1,  # Finger + Speech
        'description': 'Two modality: Finger + Speech'
    },
    'exp7_all_real': {
        'model_subset_choice': 0,  # All three
        'description': 'Three modality: All real (Gold standard)'
    }
}

# Default configuration (from wandb sweeps or manual setting)
DEFAULT_CFG = {
    'seed': 42,
    'minority_oversample': 'no',
    'sampler': 'SMOTE',
    'random_state': 42,
    'dropout_p': 0.3,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'batch_size': 32,
    'epochs': 100
}

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True,
                       choices=list(EXPERIMENTS.keys()),
                       help='Which experiment to run')
    args = parser.parse_args()
    
    exp_config = EXPERIMENTS[args.experiment]
    print(f"\n{'='*80}")
    print(f"Running: {args.experiment}")
    print(f"Description: {exp_config['description']}")
    print(f"Model subset: {exp_config['model_subset_choice']}")
    print(f"{'='*80}\n")
    
    # Merge configs
    cfg = {**DEFAULT_CFG, **exp_config}
    
    # Run experiment
    main(**cfg)
    
    print(f"\n✅ Experiment {args.experiment} completed!\n")


