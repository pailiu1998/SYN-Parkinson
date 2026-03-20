#!/usr/bin/env python
"""
计算 Scenario 6 vs Original Fusion 的统计显著性 (p-value)

使用配对 t 检验 (paired t-test) 来比较两个模型在相同随机种子下的性能差异
"""

import json
import numpy as np
from scipy import stats
import os
from pathlib import Path

# 配置
SCENARIO6_DIR = "scenario_6_results_20251215_083315"
ORIGINAL_DIR = "original_fusion_results"

def load_results(results_dir, metric_type='test'):
    """
    从结果目录加载所有 seeds 的结果
    
    Args:
        results_dir: 结果目录路径
        metric_type: 'test' 或 'dev'
    
    Returns:
        dict: {seed: metrics_dict}
    """
    results = {}
    
    if "scenario_6" in results_dir:
        # Scenario 6 使用平铺的文件格式
        pattern = f"{metric_type}_seed_*.json"
        for json_file in Path(results_dir).glob(pattern):
            seed = int(json_file.stem.split('_')[-1])
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    results[seed] = data
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
    else:
        # Original 使用 seed_X/ 目录格式
        for seed_dir in Path(results_dir).glob("seed_*"):
            seed = int(seed_dir.name.split('_')[1])
            json_file = seed_dir / f"fusion_model_results_{metric_type}.json"
            
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        results[seed] = data
                except Exception as e:
                    print(f"Warning: Failed to load {json_file}: {e}")
    
    return results

def extract_metric_arrays(scenario6_results, original_results, metric_name):
    """
    提取指定指标的配对数组
    
    Args:
        scenario6_results: Scenario 6 结果字典
        original_results: Original 结果字典
        metric_name: 指标名称 (如 'f1', 'auroc')
    
    Returns:
        tuple: (scenario6_values, original_values, common_seeds)
    """
    # 找到共同的 seeds
    common_seeds = sorted(set(scenario6_results.keys()) & set(original_results.keys()))
    
    scenario6_values = []
    original_values = []
    
    for seed in common_seeds:
        try:
            s6_val = scenario6_results[seed].get(metric_name)
            orig_val = original_results[seed].get(metric_name)
            
            if s6_val is not None and orig_val is not None:
                scenario6_values.append(s6_val)
                original_values.append(orig_val)
        except Exception as e:
            print(f"Warning: Failed to extract {metric_name} for seed {seed}: {e}")
            continue
    
    return np.array(scenario6_values), np.array(original_values), common_seeds

def perform_paired_ttest(scenario6_values, original_values):
    """
    执行配对 t 检验
    
    Args:
        scenario6_values: Scenario 6 的指标值数组
        original_values: Original 的指标值数组
    
    Returns:
        dict: 包含统计信息的字典
    """
    # 配对 t 检验
    t_statistic, p_value = stats.ttest_rel(scenario6_values, original_values)
    
    # 计算均值和标准差
    s6_mean = np.mean(scenario6_values)
    s6_std = np.std(scenario6_values, ddof=1)
    orig_mean = np.mean(original_values)
    orig_std = np.std(original_values, ddof=1)
    
    # 计算差异
    differences = scenario6_values - original_values
    diff_mean = np.mean(differences)
    diff_std = np.std(differences, ddof=1)
    
    # 计算效应量 (Cohen's d)
    cohens_d = diff_mean / diff_std if diff_std > 0 else 0
    
    # 95% 置信区间
    n = len(scenario6_values)
    se = diff_std / np.sqrt(n)
    t_critical = stats.t.ppf(0.975, n - 1)
    ci_lower = diff_mean - t_critical * se
    ci_upper = diff_mean + t_critical * se
    
    return {
        'n': n,
        'scenario6_mean': s6_mean,
        'scenario6_std': s6_std,
        'original_mean': orig_mean,
        'original_std': orig_std,
        'difference_mean': diff_mean,
        'difference_std': diff_std,
        'difference_ci_95': [ci_lower, ci_upper],
        't_statistic': t_statistic,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant_at_0.05': p_value < 0.05,
        'significant_at_0.01': p_value < 0.01,
        'significant_at_0.001': p_value < 0.001
    }

def main():
    print("="*80)
    print("Scenario 6 vs Original Fusion - Statistical Significance Test")
    print("="*80)
    print()
    
    # 加载结果
    print("Loading results...")
    scenario6_test = load_results(SCENARIO6_DIR, 'test')
    scenario6_dev = load_results(SCENARIO6_DIR, 'dev')
    original_test = load_results(ORIGINAL_DIR, 'test')
    original_dev = load_results(ORIGINAL_DIR, 'dev')
    
    print(f"Scenario 6 Test: {len(scenario6_test)} seeds")
    print(f"Scenario 6 Dev:  {len(scenario6_dev)} seeds")
    print(f"Original Test:   {len(original_test)} seeds")
    print(f"Original Dev:    {len(original_dev)} seeds")
    print()
    
    # 定义要比较的指标
    metrics = {
        'f1': 'F1 Score',
        'auroc': 'AUROC',
        'balanced_accuracy': 'Balanced Accuracy',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall'
    }
    
    # 存储所有结果
    all_results = {
        'test_metrics': {},
        'dev_metrics': {}
    }
    
    # 分析 Test Set
    print("="*80)
    print("TEST SET ANALYSIS")
    print("="*80)
    print()
    
    for metric_key, metric_name in metrics.items():
        s6_vals, orig_vals, common_seeds = extract_metric_arrays(
            scenario6_test, original_test, metric_key
        )
        
        if len(s6_vals) < 2:
            print(f"⚠️  {metric_name}: Insufficient data (n={len(s6_vals)})")
            continue
        
        result = perform_paired_ttest(s6_vals, orig_vals)
        all_results['test_metrics'][metric_key] = result
        
        # 打印结果
        print(f"{metric_name}:")
        print(f"  Sample size (n):        {result['n']}")
        print(f"  Scenario 6 mean:        {result['scenario6_mean']:.4f} ± {result['scenario6_std']:.4f}")
        print(f"  Original mean:          {result['original_mean']:.4f} ± {result['original_std']:.4f}")
        print(f"  Difference:             {result['difference_mean']:.4f} ± {result['difference_std']:.4f}")
        print(f"  95% CI of difference:   [{result['difference_ci_95'][0]:.4f}, {result['difference_ci_95'][1]:.4f}]")
        print(f"  t-statistic:            {result['t_statistic']:.4f}")
        print(f"  p-value:                {result['p_value']:.6f}", end='')
        
        if result['p_value'] < 0.001:
            print(" ***")
        elif result['p_value'] < 0.01:
            print(" **")
        elif result['p_value'] < 0.05:
            print(" *")
        else:
            print(" (n.s.)")
        
        print(f"  Cohen's d:              {result['cohens_d']:.4f}")
        print()
    
    # 分析 Dev Set
    print("="*80)
    print("DEV SET ANALYSIS")
    print("="*80)
    print()
    
    for metric_key, metric_name in metrics.items():
        s6_vals, orig_vals, common_seeds = extract_metric_arrays(
            scenario6_dev, original_dev, metric_key
        )
        
        if len(s6_vals) < 2:
            print(f"⚠️  {metric_name}: Insufficient data (n={len(s6_vals)})")
            continue
        
        result = perform_paired_ttest(s6_vals, orig_vals)
        all_results['dev_metrics'][metric_key] = result
        
        # 打印结果
        print(f"{metric_name}:")
        print(f"  Sample size (n):        {result['n']}")
        print(f"  Scenario 6 mean:        {result['scenario6_mean']:.4f} ± {result['scenario6_std']:.4f}")
        print(f"  Original mean:          {result['original_mean']:.4f} ± {result['original_std']:.4f}")
        print(f"  Difference:             {result['difference_mean']:.4f} ± {result['difference_std']:.4f}")
        print(f"  95% CI of difference:   [{result['difference_ci_95'][0]:.4f}, {result['difference_ci_95'][1]:.4f}]")
        print(f"  t-statistic:            {result['t_statistic']:.4f}")
        print(f"  p-value:                {result['p_value']:.6f}", end='')
        
        if result['p_value'] < 0.001:
            print(" ***")
        elif result['p_value'] < 0.01:
            print(" **")
        elif result['p_value'] < 0.05:
            print(" *")
        else:
            print(" (n.s.)")
        
        print(f"  Cohen's d:              {result['cohens_d']:.4f}")
        print()
    
    # 保存结果到 JSON
    output_file = "comparison_scenario6_vs_original_pvalues.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print("="*80)
    print(f"Results saved to: {output_file}")
    print("="*80)
    print()
    print("Significance levels:")
    print("  *** p < 0.001 (highly significant)")
    print("  **  p < 0.01  (very significant)")
    print("  *   p < 0.05  (significant)")
    print("  n.s. = not significant")
    print()
    print("Cohen's d interpretation:")
    print("  |d| < 0.2:  small effect")
    print("  |d| < 0.5:  medium effect")
    print("  |d| >= 0.5: large effect")
    print()

if __name__ == "__main__":
    main()

