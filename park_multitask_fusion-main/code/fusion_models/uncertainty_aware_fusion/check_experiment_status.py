#!/usr/bin/env python
"""
Check status of running experiments
"""
from pathlib import Path
import re

RESULTS_DIR = Path("/localdisk2/pliu/park_multitask_fusion-main/results/synthetic_validation/vae_diffusion_comparison")

EXPERIMENTS = [
    'vae_exp1_smile_quick_real_finger_synth',
    'vae_exp2_smile_finger_real_quick_synth',
    'vae_exp3_quick_finger_real_smile_synth',
    'diff_exp1_smile_quick_real_finger_synth',
    'diff_exp2_smile_finger_real_quick_synth',
    'diff_exp3_quick_finger_real_smile_synth'
]

def extract_auroc(log_file):
    """Extract AUROC from log file"""
    if not log_file.exists():
        return None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Look for AUROC in various formats
    patterns = [
        r"'auroc':\s*([\d.]+)",
        r'"auroc":\s*([\d.]+)',
        r'auroc.*?([\d.]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])  # Get last match
            except:
                pass
    
    return None

def check_status():
    print("="*80)
    print("EXPERIMENT STATUS CHECK")
    print("="*80)
    
    results = {}
    for exp_name in EXPERIMENTS:
        log_file = RESULTS_DIR / f"{exp_name}.log"
        auroc = extract_auroc(log_file)
        
        if log_file.exists():
            size = log_file.stat().st_size / 1024  # KB
            if auroc is not None:
                status = f"✅ Completed (AUROC: {auroc:.4f})"
            elif size > 100:  # Log file has content
                status = "🔄 Running..."
            else:
                status = "⏳ Starting..."
        else:
            status = "❌ Not started"
        
        results[exp_name] = {'status': status, 'auroc': auroc}
        print(f"{exp_name:50s}: {status}")
    
    print("="*80)
    
    # Summary
    completed = sum(1 for r in results.values() if r['auroc'] is not None)
    print(f"\nSummary: {completed}/{len(EXPERIMENTS)} experiments completed")
    
    if completed > 0:
        print("\nCompleted experiments:")
        for exp_name, result in results.items():
            if result['auroc'] is not None:
                print(f"  {exp_name}: AUROC = {result['auroc']:.4f}")

if __name__ == '__main__':
    check_status()


