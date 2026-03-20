#!/usr/bin/env python
"""
Run scenario 4 with multiple random seeds and compute confidence intervals
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import subprocess
import sys
from datetime import datetime

def run_experiment(seed):
    """Run a single experiment with given seed"""
    print(f"\n{'='*60}")
    print(f"Running experiment with seed: {seed}")
    print(f"{'='*60}\n")
    
    # Run the uncertainty_aware_fusion_scenario_4.py with the given seed
    cmd = [
        sys.executable,
        "uncertainty_aware_fusion_scenario_4.py",
        f"--seed={seed}"
    ]
    
    try:
        # Run without capturing output to see real-time progress
        result = subprocess.run(
            cmd,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            print(f"\n{'='*60}")
            print(f"✓ Seed {seed} completed successfully")
            print(f"{'='*60}\n")
            return True
        else:
            print(f"\n{'='*60}")
            print(f"✗ Seed {seed} failed with return code: {result.returncode}")
            print(f"{'='*60}\n")
            return False
    except subprocess.TimeoutExpired:
        print(f"\n{'='*60}")
        print(f"✗ Seed {seed} timed out")
        print(f"{'='*60}\n")
        return False
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ Seed {seed} failed with exception: {e}")
        print(f"{'='*60}\n")
        return False

def collect_results():
    """Collect results from all JSON files"""
    results = {
        'test': [],
        'dev': []
    }
    
    # Collect test results
    if os.path.exists('fusion_model_results_test.json'):
        with open('fusion_model_results_test.json', 'r') as f:
            data = json.load(f)
            results['test'].append(data)
    
    # Collect dev results  
    if os.path.exists('fusion_model_results_dev.json'):
        with open('fusion_model_results_dev.json', 'r') as f:
            data = json.load(f)
            results['dev'].append(data)
    
    return results

def compute_metrics_from_predictions(predictions, labels):
    """Compute metrics from predictions and labels"""
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, f1_score, 
        precision_score, recall_score, confusion_matrix,
        average_precision_score, brier_score_loss
    )
    
    preds = np.array(predictions)
    labs = np.array(labels)
    
    # Binary predictions
    binary_preds = (preds >= 0.5).astype(int)
    
    metrics = {}
    metrics['accuracy'] = accuracy_score(labs, binary_preds)
    metrics['auroc'] = roc_auc_score(labs, preds)
    metrics['f1_score'] = f1_score(labs, binary_preds)
    metrics['precision'] = precision_score(labs, binary_preds, zero_division=0)
    metrics['recall'] = recall_score(labs, binary_preds, zero_division=0)
    metrics['average_precision'] = average_precision_score(labs, preds)
    metrics['brier_score'] = brier_score_loss(labs, preds)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labs, binary_preds).ravel()
    metrics['tn'] = int(tn)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)
    metrics['tp'] = int(tp)
    
    # Weighted accuracy
    if (tp + fp) > 0 and (tn + fn) > 0:
        metrics['weighted_accuracy'] = (tp/(tp+fp) + tn/(tn+fn)) / 2.0
    else:
        metrics['weighted_accuracy'] = 0.0
    
    # Sensitivity and Specificity
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return metrics

def compute_confidence_interval(data, confidence=0.95):
    """Compute mean and confidence interval"""
    n = len(data)
    if n == 0:
        return None, None, None
    
    mean = np.mean(data)
    std_err = stats.sem(data)
    
    # Use t-distribution for confidence interval
    ci = stats.t.interval(confidence, n-1, loc=mean, scale=std_err)
    
    return mean, ci[0], ci[1]

def main():
    # Create directory for storing individual results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"scenario_4_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Running Scenario 4 with seeds 0-100")
    print(f"Results will be saved to: {results_dir}")
    print(f"{'='*70}\n")
    
    all_test_metrics = []
    all_dev_metrics = []
    successful_seeds = []
    failed_seeds = []
    
    # Run experiments with different seeds
    for seed in range(101):  # 0 to 100
        success = run_experiment(seed)
        
        if success:
            # Collect and save results
            results = collect_results()
            
            # Process test results
            if results['test']:
                test_data = results['test'][0]
                test_metrics = compute_metrics_from_predictions(
                    test_data['prediction'], 
                    test_data['label']
                )
                test_metrics['seed'] = seed
                all_test_metrics.append(test_metrics)
                
                # Save individual result
                with open(f"{results_dir}/test_seed_{seed}.json", 'w') as f:
                    json.dump(test_metrics, f, indent=2)
            
            # Process dev results
            if results['dev']:
                dev_data = results['dev'][0]
                dev_metrics = compute_metrics_from_predictions(
                    dev_data['prediction'],
                    dev_data['label']
                )
                dev_metrics['seed'] = seed
                all_dev_metrics.append(dev_metrics)
                
                # Save individual result
                with open(f"{results_dir}/dev_seed_{seed}.json", 'w') as f:
                    json.dump(dev_metrics, f, indent=2)
            
            successful_seeds.append(seed)
        else:
            failed_seeds.append(seed)
    
    # Compute statistics across all seeds
    print(f"\n{'='*70}")
    print(f"Computing statistics across {len(successful_seeds)} successful runs")
    print(f"{'='*70}\n")
    
    # Convert to DataFrame for easier processing
    test_df = pd.DataFrame(all_test_metrics)
    dev_df = pd.DataFrame(all_dev_metrics)
    
    # Compute confidence intervals for each metric
    metric_names = ['accuracy', 'auroc', 'f1_score', 'precision', 'recall', 
                    'average_precision', 'brier_score', 'weighted_accuracy',
                    'sensitivity', 'specificity']
    
    # Test set statistics
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)
    test_stats = {}
    for metric in metric_names:
        if metric in test_df.columns:
            mean, ci_lower, ci_upper = compute_confidence_interval(test_df[metric])
            test_stats[metric] = {
                'mean': float(mean),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'std': float(np.std(test_df[metric])),
                'min': float(np.min(test_df[metric])),
                'max': float(np.max(test_df[metric]))
            }
            print(f"{metric:25s}: {mean:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    
    # Dev set statistics
    print("\n" + "="*70)
    print("DEV SET RESULTS")
    print("="*70)
    dev_stats = {}
    for metric in metric_names:
        if metric in dev_df.columns:
            mean, ci_lower, ci_upper = compute_confidence_interval(dev_df[metric])
            dev_stats[metric] = {
                'mean': float(mean),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'std': float(np.std(dev_df[metric])),
                'min': float(np.min(dev_df[metric])),
                'max': float(np.max(dev_df[metric]))
            }
            print(f"{metric:25s}: {mean:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    
    # Save summary statistics
    summary = {
        'timestamp': timestamp,
        'total_seeds': 101,
        'successful_seeds': successful_seeds,
        'failed_seeds': failed_seeds,
        'num_successful': len(successful_seeds),
        'num_failed': len(failed_seeds),
        'test_statistics': test_stats,
        'dev_statistics': dev_stats
    }
    
    with open(f"{results_dir}/summary_statistics.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    test_df.to_csv(f"{results_dir}/test_all_seeds.csv", index=False)
    dev_df.to_csv(f"{results_dir}/dev_all_seeds.csv", index=False)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {results_dir}")
    print(f"Successful runs: {len(successful_seeds)}/101")
    print(f"Failed runs: {len(failed_seeds)}/101")
    print(f"{'='*70}\n")
    
    if failed_seeds:
        print(f"Failed seeds: {failed_seeds}")

if __name__ == "__main__":
    main()

