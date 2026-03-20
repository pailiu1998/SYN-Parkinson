#!/usr/bin/env python
"""
Extract and display experiment results from log files
"""
import re
from pathlib import Path
import json

RESULTS_DIR = Path("/localdisk2/pliu/park_multitask_fusion-main/results/synthetic_validation/vae_diffusion_comparison")

EXPERIMENTS = {
    'vae_exp1_smile_quick_real_finger_synth': 'VAE: Real Smile + Real Quick + Synthetic Finger',
    'vae_exp2_smile_finger_real_quick_synth': 'VAE: Real Smile + Real Finger + Synthetic Quick',
    'vae_exp3_quick_finger_real_smile_synth': 'VAE: Real Quick + Real Finger + Synthetic Smile',
    'diff_exp1_smile_quick_real_finger_synth': 'Diffusion: Real Smile + Real Quick + Synthetic Finger',
    'diff_exp2_smile_finger_real_quick_synth': 'Diffusion: Real Smile + Real Finger + Synthetic Quick',
    'diff_exp3_quick_finger_real_smile_synth': 'Diffusion: Real Quick + Real Finger + Synthetic Smile',
}

def extract_metrics(log_file):
    """Extract metrics from log file"""
    if not log_file.exists():
        return None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    metrics = {}
    
    # Extract AUROC
    auroc_patterns = [
        r"'auroc':\s*([\d.]+)",
        r'"auroc":\s*([\d.]+)',
        r'auroc.*?([\d.]+)',
        r'wandb:.*auroc\s+([\d.]+)',
    ]
    for pattern in auroc_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            try:
                metrics['auroc'] = float(matches[-1])
                break
            except:
                pass
    
    # Extract accuracy
    acc_patterns = [
        r"'accuracy':\s*([\d.]+)",
        r'"accuracy":\s*([\d.]+)',
        r'accuracy.*?([\d.]+)',
        r'wandb:.*accuracy\s+([\d.]+)',
    ]
    for pattern in acc_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            try:
                metrics['accuracy'] = float(matches[-1])
                break
            except:
                pass
    
    # Extract F1 score
    f1_patterns = [
        r"'f1_score':\s*([\d.]+)",
        r'"f1_score":\s*([\d.]+)',
        r'f1.*?([\d.]+)',
    ]
    for pattern in f1_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            try:
                metrics['f1_score'] = float(matches[-1])
                break
            except:
                pass
    
    # Check for errors
    if 'Error' in content or 'Traceback' in content or 'KeyError' in content:
        metrics['status'] = 'error'
        # Extract error message
        error_match = re.search(r'(KeyError|Error|Traceback.*?\n.*?\n.*?):\s*(.+)', content, re.DOTALL)
        if error_match:
            metrics['error'] = error_match.group(0)[:200]  # First 200 chars
    elif metrics:
        metrics['status'] = 'completed'
    else:
        metrics['status'] = 'running'
    
    return metrics

def main():
    print("="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print()
    
    results = {}
    for exp_name, exp_desc in EXPERIMENTS.items():
        log_file = RESULTS_DIR / f"{exp_name}.log"
        metrics = extract_metrics(log_file)
        results[exp_name] = metrics
        
        print(f"{exp_name}")
        print(f"  Description: {exp_desc}")
        if metrics:
            if metrics.get('status') == 'error':
                print(f"  Status: ❌ Error")
                if 'error' in metrics:
                    print(f"  Error: {metrics['error'][:100]}...")
            elif metrics.get('status') == 'completed':
                print(f"  Status: ✅ Completed")
                if 'auroc' in metrics:
                    print(f"  AUROC: {metrics['auroc']:.4f}")
                if 'accuracy' in metrics:
                    print(f"  Accuracy: {metrics['accuracy']:.4f}")
                if 'f1_score' in metrics:
                    print(f"  F1 Score: {metrics['f1_score']:.4f}")
            else:
                print(f"  Status: ⏳ Running...")
        else:
            print(f"  Status: ❌ No log file found")
        print()
    
    # Summary table
    print("="*80)
    print("RESULTS TABLE")
    print("="*80)
    print(f"{'Experiment':<50} {'Status':<15} {'AUROC':<10} {'Accuracy':<10}")
    print("-"*80)
    
    for exp_name, metrics in results.items():
        if metrics:
            status = metrics.get('status', 'unknown')
            auroc = f"{metrics.get('auroc', 0):.4f}" if 'auroc' in metrics else "-"
            accuracy = f"{metrics.get('accuracy', 0):.4f}" if 'accuracy' in metrics else "-"
            print(f"{exp_name:<50} {status:<15} {auroc:<10} {accuracy:<10}")
        else:
            print(f"{exp_name:<50} {'no log':<15} {'-':<10} {'-':<10}")
    
    # Save results to JSON
    output_file = RESULTS_DIR / "results_summary.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print()
    print(f"✅ Results saved to: {output_file}")

if __name__ == '__main__':
    main()


