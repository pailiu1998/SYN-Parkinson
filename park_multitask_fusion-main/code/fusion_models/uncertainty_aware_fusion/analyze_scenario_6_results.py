#!/usr/bin/env python
"""
Analyze results from multiple seed runs and compute confidence intervals
"""
import os
import json
import glob
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    average_precision_score, brier_score_loss
)
import argparse

def compute_metrics_from_predictions(predictions, labels):
    """Compute metrics from predictions and labels"""
    preds = np.array(predictions)
    labs = np.array(labels)
    
    # Binary predictions
    binary_preds = (preds >= 0.5).astype(int)
    
    metrics = {}
    metrics['accuracy'] = accuracy_score(labs, binary_preds)
    
    try:
        metrics['auroc'] = roc_auc_score(labs, preds)
    except:
        metrics['auroc'] = np.nan
    
    metrics['f1_score'] = f1_score(labs, binary_preds, zero_division=0)
    metrics['precision'] = precision_score(labs, binary_preds, zero_division=0)
    metrics['recall'] = recall_score(labs, binary_preds, zero_division=0)
    
    try:
        metrics['average_precision'] = average_precision_score(labs, preds)
    except:
        metrics['average_precision'] = np.nan
    
    metrics['brier_score'] = brier_score_loss(labs, preds)
    
    # Confusion matrix
    try:
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
    except:
        metrics['weighted_accuracy'] = np.nan
        metrics['sensitivity'] = np.nan
        metrics['specificity'] = np.nan
    
    return metrics

def compute_confidence_interval(data, confidence=0.95):
    """Compute mean and confidence interval"""
    data = np.array(data)
    data = data[~np.isnan(data)]  # Remove NaN values
    
    n = len(data)
    if n == 0:
        return np.nan, np.nan, np.nan
    
    mean = np.mean(data)
    
    if n == 1:
        return mean, mean, mean
    
    std_err = stats.sem(data)
    
    # Use t-distribution for confidence interval
    ci = stats.t.interval(confidence, n-1, loc=mean, scale=std_err)
    
    return mean, ci[0], ci[1]

def analyze_results(results_dir, output_file=None):
    """Analyze results from a directory"""
    
    # Find all result files
    test_files = sorted(glob.glob(f"{results_dir}/test_seed_*.json"))
    dev_files = sorted(glob.glob(f"{results_dir}/dev_seed_*.json"))
    
    print(f"\nFound {len(test_files)} test result files")
    print(f"Found {len(dev_files)} dev result files")
    
    # Process test results
    test_metrics_list = []
    for test_file in test_files:
        seed = int(test_file.split('seed_')[1].split('.')[0])
        with open(test_file, 'r') as f:
            data = json.load(f)
            
        # Check if data has predictions and labels
        if 'prediction' in data and 'label' in data:
            metrics = compute_metrics_from_predictions(data['prediction'], data['label'])
        else:
            # Assume the file already contains computed metrics
            metrics = data
        
        metrics['seed'] = seed
        test_metrics_list.append(metrics)
    
    # Process dev results
    dev_metrics_list = []
    for dev_file in dev_files:
        seed = int(dev_file.split('seed_')[1].split('.')[0])
        with open(dev_file, 'r') as f:
            data = json.load(f)
            
        # Check if data has predictions and labels
        if 'prediction' in data and 'label' in data:
            metrics = compute_metrics_from_predictions(data['prediction'], data['label'])
        else:
            metrics = data
        
        metrics['seed'] = seed
        dev_metrics_list.append(metrics)
    
    # Convert to DataFrames
    test_df = pd.DataFrame(test_metrics_list)
    dev_df = pd.DataFrame(dev_metrics_list)
    
    # Compute statistics
    metric_names = ['accuracy', 'auroc', 'f1_score', 'precision', 'recall',
                    'average_precision', 'brier_score', 'weighted_accuracy',
                    'sensitivity', 'specificity']
    
    results_summary = {
        'test': {},
        'dev': {}
    }
    
    # Test statistics
    print("\n" + "="*80)
    print("TEST SET RESULTS (95% Confidence Intervals)")
    print("="*80)
    for metric in metric_names:
        if metric in test_df.columns:
            mean, ci_lower, ci_upper = compute_confidence_interval(test_df[metric])
            std = np.nanstd(test_df[metric])
            results_summary['test'][metric] = {
                'mean': float(mean) if not np.isnan(mean) else None,
                'ci_lower': float(ci_lower) if not np.isnan(ci_lower) else None,
                'ci_upper': float(ci_upper) if not np.isnan(ci_upper) else None,
                'std': float(std) if not np.isnan(std) else None,
                'min': float(np.nanmin(test_df[metric])),
                'max': float(np.nanmax(test_df[metric])),
                'n': int(np.sum(~np.isnan(test_df[metric])))
            }
            print(f"{metric:25s}: {mean:.4f} ± {std:.4f}  [95% CI: {ci_lower:.4f} - {ci_upper:.4f}]")
    
    # Dev statistics
    print("\n" + "="*80)
    print("DEV SET RESULTS (95% Confidence Intervals)")
    print("="*80)
    for metric in metric_names:
        if metric in dev_df.columns:
            mean, ci_lower, ci_upper = compute_confidence_interval(dev_df[metric])
            std = np.nanstd(dev_df[metric])
            results_summary['dev'][metric] = {
                'mean': float(mean) if not np.isnan(mean) else None,
                'ci_lower': float(ci_lower) if not np.isnan(ci_lower) else None,
                'ci_upper': float(ci_upper) if not np.isnan(ci_upper) else None,
                'std': float(std) if not np.isnan(std) else None,
                'min': float(np.nanmin(dev_df[metric])),
                'max': float(np.nanmax(dev_df[metric])),
                'n': int(np.sum(~np.isnan(dev_df[metric])))
            }
            print(f"{metric:25s}: {mean:.4f} ± {std:.4f}  [95% CI: {ci_lower:.4f} - {ci_upper:.4f}]")
    
    # Save summary
    summary_file = f"{results_dir}/summary_statistics.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save detailed CSVs
    test_df.to_csv(f"{results_dir}/test_all_seeds.csv", index=False)
    dev_df.to_csv(f"{results_dir}/dev_all_seeds.csv", index=False)
    
    print(f"\n{'='*80}")
    print(f"Summary saved to: {summary_file}")
    print(f"Detailed results saved to:")
    print(f"  - {results_dir}/test_all_seeds.csv")
    print(f"  - {results_dir}/dev_all_seeds.csv")
    print(f"{'='*80}\n")
    
    return results_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze scenario 6 results')
    parser.add_argument('results_dir', type=str, help='Directory containing result files')
    parser.add_argument('--output', type=str, default=None, help='Output file for summary')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Directory {args.results_dir} does not exist")
        exit(1)
    
    analyze_results(args.results_dir, args.output)


