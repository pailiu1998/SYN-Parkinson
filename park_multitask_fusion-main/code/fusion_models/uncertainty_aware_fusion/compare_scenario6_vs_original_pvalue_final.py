#!/usr/bin/env python
"""
计算 Scenario 6 vs Original Fusion的统计显著性 (p-value)
使用 CSV 文件进行配对 t 检验
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score

# 配置
SCENARIO6_TEST_CSV = "scenario_6_results_20251215_083315/test_all_seeds.csv"
SCENARIO6_DEV_CSV = "scenario_6_results_20251215_083315/dev_all_seeds.csv"

def calculate_metrics_from_predictions(pred_file):
    """从原始预测文件计算指标"""
    with open(pred_file, 'r') as f:
        data = json.load(f)
    
    predictions = np.array(data['prediction'])
    labels = np.array(data['label'])
    
    # 转换预测为二分类
    pred_binary = (predictions >= 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(labels, pred_binary),
        'f1': f1_score(labels, pred_binary),
        'precision': precision_score(labels, pred_binary, zero_division=0),
        'recall': recall_score(labels, pred_binary, zero_division=0),
        'auroc': roc_auc_score(labels, predictions),
        'balanced_accuracy': balanced_accuracy_score(labels, pred_binary)
    }
    
    return metrics

def load_original_results_from_seeds():
    """从 seed 目录加载 Original 的结果"""
    import os
    from pathlib import Path
    
    test_records = []
    dev_records = []
    
    results_dir = Path("original_fusion_results")
    
    for seed_dir in sorted(results_dir.glob("seed_*")):
        seed = int(seed_dir.name.split('_')[1])
        
        # Test set
        test_file = seed_dir / "fusion_model_results_test.json"
        if test_file.exists():
            try:
                metrics = calculate_metrics_from_predictions(test_file)
                metrics['seed'] = seed
                test_records.append(metrics)
            except Exception as e:
                print(f"Warning: Failed to process {test_file}: {e}")
        
        # Dev set
        dev_file = seed_dir / "fusion_model_results_dev.json"
        if dev_file.exists():
            try:
                metrics = calculate_metrics_from_predictions(dev_file)
                metrics['seed'] = seed
                dev_records.append(metrics)
            except Exception as e:
                print(f"Warning: Failed to process {dev_file}: {e}")
    
    test_df = pd.DataFrame(test_records)
    dev_df = pd.DataFrame(dev_records)
    
    return test_df, dev_df

def perform_paired_ttest(scenario6_values, original_values):
    """执行配对 t 检验"""
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
        'n': int(n),
        'scenario6_mean': float(s6_mean),
        'scenario6_std': float(s6_std),
        'original_mean': float(orig_mean),
        'original_std': float(orig_std),
        'difference_mean': float(diff_mean),
        'difference_std': float(diff_std),
        'difference_ci_95': [float(ci_lower), float(ci_upper)],
        't_statistic': float(t_statistic),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'significant_at_0.05': bool(p_value < 0.05),
        'significant_at_0.01': bool(p_value < 0.01),
        'significant_at_0.001': bool(p_value < 0.001)
    }

def main():
    print("="*80)
    print("Scenario 6 vs Original Fusion - Statistical Significance Test")
    print("="*80)
    print()
    
    # 加载结果
    print("Loading Scenario 6 results...")
    scenario6_test = pd.read_csv(SCENARIO6_TEST_CSV)
    scenario6_dev = pd.read_csv(SCENARIO6_DEV_CSV)
    
    print("Loading Original results (calculating from predictions)...")
    original_test, original_dev = load_original_results_from_seeds()
    
    print(f"\nScenario 6 Test: {len(scenario6_test)} seeds")
    print(f"Scenario 6 Dev:  {len(scenario6_dev)} seeds")
    print(f"Original Test:   {len(original_test)} seeds")
    print(f"Original Dev:    {len(original_dev)} seeds")
    print()
    
    # 定义指标映射
    metrics = {
        'f1_score': ('f1', 'F1 Score'),
        'auroc': ('auroc', 'AUROC'),
        'weighted_accuracy': ('balanced_accuracy', 'Balanced Accuracy'),
        'accuracy': ('accuracy', 'Accuracy'),
        'precision': ('precision', 'Precision'),
        'recall': ('recall', 'Recall')
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
    
    # 合并数据框（基于 seed）
    test_merged = pd.merge(
        scenario6_test,
        original_test,
        on='seed',
        suffixes=('_s6', '_orig')
    )
    
    print(f"Common seeds in test set: {len(test_merged)}")
    print()
    
    for s6_col, (orig_col, display_name) in metrics.items():
        # 查找列名
        s6_col_name = s6_col if s6_col in test_merged.columns else s6_col + '_s6'
        orig_col_name = orig_col if orig_col in test_merged.columns else orig_col + '_orig'
        
        if s6_col_name not in test_merged.columns or orig_col_name not in test_merged.columns:
            print(f"⚠️  {display_name}: Column not found")
            print(f"    Available Scenario 6 columns: {[c for c in test_merged.columns if '_s6' in c or c == s6_col]}")
            print(f"    Available Original columns: {[c for c in test_merged.columns if '_orig' in c or c == orig_col]}")
            continue
        
        s6_vals = test_merged[s6_col_name].values
        orig_vals = test_merged[orig_col_name].values
        
        # 移除 NaN 值
        mask = ~(np.isnan(s6_vals) | np.isnan(orig_vals))
        s6_vals = s6_vals[mask]
        orig_vals = orig_vals[mask]
        
        if len(s6_vals) < 2:
            print(f"⚠️  {display_name}: Insufficient data (n={len(s6_vals)})")
            continue
        
        result = perform_paired_ttest(s6_vals, orig_vals)
        all_results['test_metrics'][orig_col] = result
        
        # 打印结果
        print(f"{display_name}:")
        print(f"  Sample size (n):        {result['n']}")
        print(f"  Scenario 6 mean:        {result['scenario6_mean']:.4f} ± {result['scenario6_std']:.4f}")
        print(f"  Original mean:          {result['original_mean']:.4f} ± {result['original_std']:.4f}")
        print(f"  Difference (S6-Orig):   {result['difference_mean']:.4f} ± {result['difference_std']:.4f}")
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
        
        # 解释效应量
        abs_d = abs(result['cohens_d'])
        if abs_d < 0.2:
            effect_size = "small"
        elif abs_d < 0.5:
            effect_size = "medium"
        else:
            effect_size = "large"
        print(f"  Effect size:            {effect_size}")
        print()
    
    # 分析 Dev Set
    print("="*80)
    print("DEV SET ANALYSIS")
    print("="*80)
    print()
    
    # 合并数据框（基于 seed）
    dev_merged = pd.merge(
        scenario6_dev,
        original_dev,
        on='seed',
        suffixes=('_s6', '_orig')
    )
    
    print(f"Common seeds in dev set: {len(dev_merged)}")
    print()
    
    for s6_col, (orig_col, display_name) in metrics.items():
        s6_col_name = s6_col if s6_col in dev_merged.columns else s6_col + '_s6'
        orig_col_name = orig_col if orig_col in dev_merged.columns else orig_col + '_orig'
        
        if s6_col_name not in dev_merged.columns or orig_col_name not in dev_merged.columns:
            print(f"⚠️  {display_name}: Column not found")
            continue
        
        s6_vals = dev_merged[s6_col_name].values
        orig_vals = dev_merged[orig_col_name].values
        
        # 移除 NaN 值
        mask = ~(np.isnan(s6_vals) | np.isnan(orig_vals))
        s6_vals = s6_vals[mask]
        orig_vals = orig_vals[mask]
        
        if len(s6_vals) < 2:
            print(f"⚠️  {display_name}: Insufficient data (n={len(s6_vals)})")
            continue
        
        result = perform_paired_ttest(s6_vals, orig_vals)
        all_results['dev_metrics'][orig_col] = result
        
        # 打印结果
        print(f"{display_name}:")
        print(f"  Sample size (n):        {result['n']}")
        print(f"  Scenario 6 mean:        {result['scenario6_mean']:.4f} ± {result['scenario6_std']:.4f}")
        print(f"  Original mean:          {result['original_mean']:.4f} ± {result['original_std']:.4f}")
        print(f"  Difference (S6-Orig):   {result['difference_mean']:.4f} ± {result['difference_std']:.4f}")
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
        
        # 解释效应量
        abs_d = abs(result['cohens_d'])
        if abs_d < 0.2:
            effect_size = "small"
        elif abs_d < 0.5:
            effect_size = "medium"
        else:
            effect_size = "large"
        print(f"  Effect size:            {effect_size}")
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
    print("Interpretation:")
    print("  - Positive difference (S6-Orig > 0): Scenario 6 performs better")
    print("  - Negative difference (S6-Orig < 0): Original performs better")
    print()

if __name__ == "__main__":
    main()

