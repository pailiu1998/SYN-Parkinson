#!/usr/bin/env python
"""
计算 Scenario 6 vs Original Fusion 的统计显著性 (p-value)
使用汇总的 CSV 文件进行配对 t 检验
"""

import pandas as pd
import numpy as np
from scipy import stats
import json

# 配置
SCENARIO6_TEST_CSV = "scenario_6_results_20251215_083315/test_all_seeds.csv"
SCENARIO6_DEV_CSV = "scenario_6_results_20251215_083315/dev_all_seeds.csv"
ORIGINAL_RESULTS_JSON = "original_fusion_results/all_seeds_results.json"

def load_scenario6_results(csv_file):
    """加载 Scenario 6 的 CSV 结果"""
    df = pd.read_csv(csv_file)
    return df

def load_original_results(json_file):
    """加载 Original 的 JSON 结果并转换为 DataFrame"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 转换为 DataFrame
    test_records = []
    dev_records = []
    
    for seed_str, metrics in data.items():
        seed = int(seed_str.replace('seed_', ''))
        
        if 'test' in metrics:
            test_rec = {'seed': seed}
            test_rec.update(metrics['test'])
            test_records.append(test_rec)
        
        if 'dev' in metrics:
            dev_rec = {'seed': seed}
            dev_rec.update(metrics['dev'])
            dev_records.append(dev_rec)
    
    test_df = pd.DataFrame(test_records)
    dev_df = pd.DataFrame(dev_records)
    
    return test_df, dev_df

def perform_paired_ttest(scenario6_values, original_values):
    """
    执行配对 t 检验
    
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
    print("Loading results...")
    scenario6_test = load_scenario6_results(SCENARIO6_TEST_CSV)
    scenario6_dev = load_scenario6_results(SCENARIO6_DEV_CSV)
    original_test, original_dev = load_original_results(ORIGINAL_RESULTS_JSON)
    
    print(f"Scenario 6 Test: {len(scenario6_test)} seeds")
    print(f"Scenario 6 Dev:  {len(scenario6_dev)} seeds")
    print(f"Original Test:   {len(original_test)} seeds")
    print(f"Original Dev:    {len(original_dev)} seeds")
    print()
    
    # 定义指标映射 (scenario 6 列名 -> original 列名)
    metric_mapping = {
        'f1_score': ('f1_score', 'f1', 'F1 Score'),
        'auroc': ('auroc', 'auroc', 'AUROC'),
        'weighted_accuracy': ('weighted_accuracy', 'balanced_accuracy', 'Balanced Accuracy'),
        'accuracy': ('accuracy', 'accuracy', 'Accuracy'),
        'precision': ('precision', 'precision', 'Precision'),
        'recall': ('recall', 'recall', 'Recall')
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
    
    for s6_col, (s6_key, orig_key, display_name) in metric_mapping.items():
        s6_col_name = s6_col if s6_col in test_merged.columns else s6_col + '_s6'
        orig_col_name = orig_key if orig_key in test_merged.columns else orig_key + '_orig'
        
        if s6_col_name not in test_merged.columns or orig_col_name not in test_merged.columns:
            print(f"⚠️  {display_name}: Column not found")
            print(f"    Looking for: {s6_col_name} and {orig_col_name}")
            print(f"    Available: {test_merged.columns.tolist()}")
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
        all_results['test_metrics'][s6_key] = result
        
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
    
    for s6_col, (s6_key, orig_key, display_name) in metric_mapping.items():
        s6_col_name = s6_col if s6_col in dev_merged.columns else s6_col + '_s6'
        orig_col_name = orig_key if orig_key in dev_merged.columns else orig_key + '_orig'
        
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
        all_results['dev_metrics'][s6_key] = result
        
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

