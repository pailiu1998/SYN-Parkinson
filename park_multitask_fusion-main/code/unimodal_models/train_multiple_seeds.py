#!/usr/bin/env python
"""
批量训练单模态模型（多个 seeds）并收集结果

使用方法:
    python train_multiple_seeds.py --modality finger --start_seed 0 --end_seed 100
    python train_multiple_seeds.py --modality quick --start_seed 0 --end_seed 100
    python train_multiple_seeds.py --modality smile --start_seed 0 --end_seed 100
"""

import os
import sys
import json
import subprocess
import argparse
import numpy as np
from scipy import stats
from pathlib import Path

def run_training(modality, seed, config):
    """运行单个模态的训练"""
    
    base_path = "/localdisk2/pliu/park_multitask_fusion-main/code/unimodal_models"
    
    # 模态配置
    modality_configs = {
        'finger': {
            'dir': 'finger_tapping',
            'script': 'unimodal_finger_baal.py',
            'params': {
                'model': 'ShallowANN',
                'dropout_prob': 0.13951215957675367,
                'num_trials': 300,
                'hand': 'both',
                'learning_rate': 0.6682837019078968,
                'batch_size': 512,
                'num_epochs': 73,
                'momentum': 0.8363833208184809,
                'use_scheduler': 'yes',
                'scheduler': 'step',
                'step_size': 22,
                'gamma': 0.6555323541714391,
            }
        },
        'quick': {
            'dir': 'quick_brown_fox',
            'script': 'unimodal_fox_baal.py',
            'params': {
                'model': 'ShallowANN',
                'dropout_prob': 0.08349938684379829,
                'num_trials': 5000,
                'learning_rate': 0.9258448866412824,
                'batch_size': 256,
                'num_epochs': 55,
                'momentum': 0.49459848722229194,
                'use_feature_scaling': 'no',
                'minority_oversample': 'yes',
                'use_scheduler': 'no',
            }
        },
        'smile': {
            'dir': 'facial_expression_smile',
            'script': 'unimodal_smile_baal.py',
            'params': {
                'model': 'ShallowANN',
                'dropout_prob': 0.1,
                'num_trials': 5000,
                'learning_rate': 0.5,
                'batch_size': 256,
                'num_epochs': 50,
                'momentum': 0.9,
                'drop_correlated': 'yes',
                'corr_thr': 0.85,
                'use_scheduler': 'yes',
                'scheduler': 'reduce',
                'patience': 10,
                'gamma': 0.5,
            }
        }
    }
    
    if modality not in modality_configs:
        raise ValueError(f"Unknown modality: {modality}")
    
    mod_config = modality_configs[modality]
    work_dir = os.path.join(base_path, mod_config['dir'])
    script = mod_config['script']
    
    # 构建命令
    cmd = ['python', script]
    
    # 添加参数
    params = mod_config['params'].copy()
    params.update(config)
    params['seed'] = seed
    params['random_state'] = 526
    
    for key, value in params.items():
        cmd.append(f'--{key}={value}')
    
    print(f"\n{'='*70}")
    print(f"训练 {modality.upper()} 模型 - Seed {seed}")
    print(f"{'='*70}")
    print(f"工作目录: {work_dir}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    # 运行训练
    try:
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            check=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败 (seed {seed}): {e}")
        return False

def collect_results(modality, results_dir):
    """收集训练结果"""
    # 这里需要根据实际的结果保存格式来实现
    # 暂时返回空
    return []

def calculate_confidence_intervals(results, confidence=0.95):
    """计算置信区间"""
    if not results:
        return {}
    
    ci_results = {}
    
    # 提取指标
    metrics = ['auroc', 'f1_score', 'accuracy', 'weighted_accuracy', 'ECE', 'loss']
    
    for metric in metrics:
        values = [r[metric] for r in results if metric in r]
        
        if values:
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            n = len(values)
            
            # 计算 t 统计量
            t_stat = stats.t.ppf((1 + confidence) / 2, n - 1)
            margin = t_stat * (std / np.sqrt(n))
            
            ci_results[metric] = {
                'mean': mean,
                'std': std,
                'ci_lower': mean - margin,
                'ci_upper': mean + margin,
                'n': n
            }
    
    return ci_results

def main():
    parser = argparse.ArgumentParser(description='批量训练单模态模型')
    parser.add_argument('--modality', type=str, required=True, 
                        choices=['finger', 'quick', 'smile'],
                        help='要训练的模态: finger, quick, smile')
    parser.add_argument('--start_seed', type=int, default=0,
                        help='起始 seed (默认: 0)')
    parser.add_argument('--end_seed', type=int, default=10,
                        help='结束 seed (默认: 10)')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='训练轮数 (覆盖默认值)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小 (覆盖默认值)')
    parser.add_argument('--results_dir', type=str, default='multi_seed_results',
                        help='结果保存目录')
    
    args = parser.parse_args()
    
    # 用户自定义配置
    custom_config = {}
    if args.num_epochs is not None:
        custom_config['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        custom_config['batch_size'] = args.batch_size
    
    # 创建结果目录
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"开始批量训练 {args.modality.upper()} 模型")
    print(f"{'='*70}")
    print(f"Seeds: {args.start_seed} - {args.end_seed}")
    print(f"总共: {args.end_seed - args.start_seed + 1} 个 seeds")
    print(f"结果目录: {results_dir}")
    print(f"{'='*70}\n")
    
    # 训练每个 seed
    successes = 0
    failures = 0
    
    for seed in range(args.start_seed, args.end_seed + 1):
        success = run_training(args.modality, seed, custom_config)
        
        if success:
            successes += 1
        else:
            failures += 1
        
        print(f"\n进度: {successes + failures}/{args.end_seed - args.start_seed + 1}")
        print(f"成功: {successes}, 失败: {failures}\n")
    
    # 收集和分析结果
    print(f"\n{'='*70}")
    print("训练完成！")
    print(f"{'='*70}")
    print(f"总共训练: {successes + failures} 个 seeds")
    print(f"成功: {successes}")
    print(f"失败: {failures}")
    
    if successes > 0:
        print(f"\n模型已保存到各自的模态目录")
        print(f"请在 wandb 查看详细结果")
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

