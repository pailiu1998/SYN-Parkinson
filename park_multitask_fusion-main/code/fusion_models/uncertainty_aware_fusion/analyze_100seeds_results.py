#!/usr/bin/env python3
"""
分析100 seeds的结果并计算置信区间
"""

import json
import numpy as np
import scipy.stats as stats
from pathlib import Path

def load_results(results_dir):
    """加载结果"""
    results_file = Path(results_dir) / "all_seeds_results.json"
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    return data

def calculate_metrics(predictions, labels):
    """计算各种评估指标"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix
    )
    
    preds = np.array(predictions)
    labs = np.array(labels)
    
    # 二值化预测
    binary_preds = (preds >= 0.5).astype(int)
    
    metrics = {}
    
    try:
        metrics['accuracy'] = accuracy_score(labs, binary_preds)
        metrics['precision'] = precision_score(labs, binary_preds, zero_division=0)
        metrics['recall'] = recall_score(labs, binary_preds, zero_division=0)
        metrics['f1'] = f1_score(labs, binary_preds, zero_division=0)
        metrics['auroc'] = roc_auc_score(labs, preds)
        
        # 计算balanced accuracy
        tn, fp, fn, tp = confusion_matrix(labs, binary_preds).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['balanced_accuracy'] = (sensitivity + specificity) / 2
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None
    
    return metrics

def compute_confidence_intervals(values, confidence=0.95):
    """计算置信区间"""
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    se = std / np.sqrt(n)
    
    # t分布
    t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_critical * se
    
    ci_lower = mean - margin
    ci_upper = mean + margin
    
    return {
        'mean': mean,
        'std': std,
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n': n
    }

def analyze_results(results_dir, scenario_name):
    """分析结果"""
    print("="*70)
    print(f"Analyzing {scenario_name}")
    print("="*70)
    
    data = load_results(results_dir)
    if data is None:
        return None
    
    # 分析test和dev集
    analysis = {}
    
    for split in ['test', 'dev']:
        print(f"\n{split.upper()} SET ANALYSIS")
        print("-"*70)
        
        split_data = data.get(split, [])
        
        if len(split_data) == 0:
            print(f"No data found for {split} set")
            continue
        
        print(f"Number of seeds: {len(split_data)}")
        
        # 计算每个seed的指标
        all_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auroc': [],
            'balanced_accuracy': []
        }
        
        for seed_result in split_data:
            predictions = seed_result.get('prediction', [])
            labels = seed_result.get('label', [])
            
            if len(predictions) == 0 or len(labels) == 0:
                continue
            
            metrics = calculate_metrics(predictions, labels)
            if metrics:
                for key in all_metrics.keys():
                    if key in metrics:
                        all_metrics[key].append(metrics[key])
        
        # 计算置信区间
        split_analysis = {}
        
        for metric_name, values in all_metrics.items():
            if len(values) > 0:
                ci_stats = compute_confidence_intervals(values)
                split_analysis[metric_name] = ci_stats
                
                print(f"\n{metric_name.upper()}:")
                print(f"  Mean:   {ci_stats['mean']:.4f}")
                print(f"  Std:    {ci_stats['std']:.4f}")
                print(f"  95% CI: [{ci_stats['ci_lower']:.4f}, {ci_stats['ci_upper']:.4f}]")
        
        analysis[split] = split_analysis
    
    # 保存分析结果
    output_file = Path(results_dir) / "statistical_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n📊 Analysis saved to: {output_file}")
    
    return analysis

def compare_scenarios(scenario1_dir, scenario2_dir, scenario1_name, scenario2_name):
    """比较两个场景的结果"""
    print("\n" + "="*70)
    print(f"COMPARISON: {scenario1_name} vs {scenario2_name}")
    print("="*70)
    
    # 加载两个场景的统计分析
    analysis1_file = Path(scenario1_dir) / "statistical_analysis.json"
    analysis2_file = Path(scenario2_dir) / "statistical_analysis.json"
    
    if not analysis1_file.exists() or not analysis2_file.exists():
        print("❌ Statistical analysis files not found. Run analyze first.")
        return
    
    with open(analysis1_file, 'r') as f:
        analysis1 = json.load(f)
    
    with open(analysis2_file, 'r') as f:
        analysis2 = json.load(f)
    
    # 比较test set结果
    print("\nTEST SET COMPARISON")
    print("-"*70)
    
    metrics_to_compare = ['f1', 'auroc', 'balanced_accuracy', 'accuracy']
    
    comparison_table = []
    comparison_table.append(f"{'Metric':<20} | {scenario1_name:<25} | {scenario2_name:<25} | {'Difference':<15}")
    comparison_table.append("-"*90)
    
    for metric in metrics_to_compare:
        if metric in analysis1.get('test', {}) and metric in analysis2.get('test', {}):
            mean1 = analysis1['test'][metric]['mean']
            ci1_lower = analysis1['test'][metric]['ci_lower']
            ci1_upper = analysis1['test'][metric]['ci_upper']
            
            mean2 = analysis2['test'][metric]['mean']
            ci2_lower = analysis2['test'][metric]['ci_lower']
            ci2_upper = analysis2['test'][metric]['ci_upper']
            
            diff = mean2 - mean1
            
            row = f"{metric.upper():<20} | {mean1:.4f} [{ci1_lower:.4f},{ci1_upper:.4f}] | {mean2:.4f} [{ci2_lower:.4f},{ci2_upper:.4f}] | {diff:+.4f}"
            comparison_table.append(row)
    
    for line in comparison_table:
        print(line)
    
    # 保存比较结果
    comparison = {
        'scenario1': scenario1_name,
        'scenario2': scenario2_name,
        'test_metrics': {},
        'dev_metrics': {}
    }
    
    for split in ['test', 'dev']:
        for metric in metrics_to_compare:
            if metric in analysis1.get(split, {}) and metric in analysis2.get(split, {}):
                comparison[f'{split}_metrics'][metric] = {
                    'scenario1_mean': analysis1[split][metric]['mean'],
                    'scenario1_ci': [analysis1[split][metric]['ci_lower'], analysis1[split][metric]['ci_upper']],
                    'scenario2_mean': analysis2[split][metric]['mean'],
                    'scenario2_ci': [analysis2[split][metric]['ci_lower'], analysis2[split][metric]['ci_upper']],
                    'difference': analysis2[split][metric]['mean'] - analysis1[split][metric]['mean']
                }
    
    output_file = "comparison_scenario10_vs_original.json"
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n📊 Comparison saved to: {output_file}")

def main():
    """主函数"""
    # 分析Scenario 10
    analyze_results("scenario_10_results", "Scenario 10 (Add Synthetic)")
    
    print("\n")
    
    # 分析Original Fusion
    analyze_results("original_fusion_results", "Original Fusion (Baseline)")
    
    # 比较两个场景
    compare_scenarios(
        "scenario_10_results",
        "original_fusion_results",
        "Scenario 10",
        "Original"
    )
    
    print("\n" + "="*70)
    print("✅ Analysis completed!")
    print("="*70)

if __name__ == "__main__":
    main()


