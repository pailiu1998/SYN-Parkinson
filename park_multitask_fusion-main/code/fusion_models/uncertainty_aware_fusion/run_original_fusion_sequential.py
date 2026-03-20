#!/usr/bin/env python3
"""
Original Fusion 顺序训练脚本 - 100 seeds
顺序运行：一次一个seed（不并行）
"""

import subprocess
import os
import json
import time
from pathlib import Path

# 配置
SCRIPT_NAME = "uncertainty_aware_fusion.py"
NUM_SEEDS = 100
START_SEED = 0
OUTPUT_DIR = "original_fusion_results"
TIMEOUT = 7200  # 2小时超时

def run_experiment(seed):
    """运行单个实验"""
    print(f"\n{'='*70}")
    print(f"[Seed {seed}/{NUM_SEEDS-1}] Starting...")
    print(f"{'='*70}")
    
    # 获取当前工作目录
    work_dir = os.getcwd()
    
    # 构建命令
    cmd = [
        "python", SCRIPT_NAME,
        "--seed", str(seed),
        "--num_epochs", "244",
        "--batch_size", "64",
        "--learning_rate", "0.001",
        "--dropout_prob", "0.25"
    ]
    
    start_time = time.time()
    
    try:
        # 在当前目录运行训练
        result = subprocess.run(
            cmd,
            text=True,
            timeout=TIMEOUT,
            cwd=work_dir
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            # 创建seed目录并移动结果文件
            seed_dir = os.path.join(OUTPUT_DIR, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            
            # 移动结果文件
            for filename in ["fusion_model_results_test.json", "fusion_model_results_dev.json"]:
                if os.path.exists(filename):
                    target = os.path.join(seed_dir, filename)
                    os.rename(filename, target)
                    print(f"✅ Moved {filename} to {seed_dir}/")
            
            print(f"[Seed {seed}] ✅ Completed in {elapsed_time:.1f}s")
            return {"seed": seed, "status": "success", "time": elapsed_time}
        else:
            print(f"[Seed {seed}] ❌ Failed with return code {result.returncode}")
            return {"seed": seed, "status": "failed", "error": f"Return code {result.returncode}"}
    
    except subprocess.TimeoutExpired:
        print(f"[Seed {seed}] ⏱️  Timeout after {TIMEOUT}s")
        return {"seed": seed, "status": "timeout", "time": TIMEOUT}
    
    except Exception as e:
        print(f"[Seed {seed}] ❌ Error: {str(e)}")
        return {"seed": seed, "status": "error", "error": str(e)}

def collect_results():
    """收集所有结果并计算统计信息"""
    print("\n" + "="*70)
    print("Collecting results from all seeds...")
    print("="*70)
    
    results = {
        'test': [],
        'dev': []
    }
    
    successful_seeds = []
    failed_seeds = []
    
    for seed in range(START_SEED, START_SEED + NUM_SEEDS):
        seed_dir = os.path.join(OUTPUT_DIR, f"seed_{seed}")
        
        # 检查test结果
        test_file = os.path.join(seed_dir, "fusion_model_results_test.json")
        if os.path.exists(test_file):
            try:
                with open(test_file, 'r') as f:
                    data = json.load(f)
                    results['test'].append(data)
                    successful_seeds.append(seed)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {test_file}: {e}")
                failed_seeds.append(seed)
        else:
            failed_seeds.append(seed)
        
        # 检查dev结果
        dev_file = os.path.join(seed_dir, "fusion_model_results_dev.json")
        if os.path.exists(dev_file):
            try:
                with open(dev_file, 'r') as f:
                    data = json.load(f)
                    results['dev'].append(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {dev_file}: {e}")
    
    # 保存汇总结果
    summary_file = os.path.join(OUTPUT_DIR, "all_seeds_results.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Successfully collected: {len(successful_seeds)} seeds")
    print(f"❌ Failed: {len(failed_seeds)} seeds")
    if failed_seeds:
        print(f"   Failed seeds: {failed_seeds[:20]}{'...' if len(failed_seeds) > 20 else ''}")
    
    print(f"\n📁 Results saved to: {summary_file}")
    
    return results

def main():
    """主函数"""
    print("="*70)
    print("Original Fusion - Sequential Training with 100 Seeds")
    print("="*70)
    print(f"Script: {SCRIPT_NAME}")
    print(f"Seeds: {START_SEED} to {START_SEED + NUM_SEEDS - 1}")
    print(f"Mode: SEQUENTIAL (one at a time)")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70 + "\n")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 记录开始时间
    total_start_time = time.time()
    
    # 顺序运行所有seeds
    run_results = []
    
    for seed in range(START_SEED, START_SEED + NUM_SEEDS):
        result = run_experiment(seed)
        run_results.append(result)
        
        # 显示进度
        completed = seed - START_SEED + 1
        total = NUM_SEEDS
        progress = (completed / total) * 100
        print(f"\n📊 Progress: {completed}/{total} ({progress:.1f}%)")
        
        # 估算剩余时间
        elapsed = time.time() - total_start_time
        avg_time = elapsed / completed
        remaining = (total - completed) * avg_time
        print(f"⏰ Elapsed: {elapsed/60:.1f} min | Estimated remaining: {remaining/60:.1f} min")
    
    # 计算总时间
    total_elapsed_time = time.time() - total_start_time
    
    # 统计运行结果
    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)
    
    successful = sum(1 for r in run_results if r.get("status") == "success")
    failed = sum(1 for r in run_results if r.get("status") == "failed")
    timeout = sum(1 for r in run_results if r.get("status") == "timeout")
    errors = sum(1 for r in run_results if r.get("status") in ["error", "exception"])
    
    print(f"Total seeds: {NUM_SEEDS}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Timeout: {timeout}")
    print(f"🔥 Errors: {errors}")
    print(f"⏰ Total time: {total_elapsed_time/3600:.2f} hours ({total_elapsed_time/60:.1f} minutes)")
    
    # 保存运行统计
    run_summary = {
        "total_seeds": NUM_SEEDS,
        "successful": successful,
        "failed": failed,
        "timeout": timeout,
        "errors": errors,
        "total_time_seconds": total_elapsed_time,
        "results": run_results
    }
    
    summary_file = os.path.join(OUTPUT_DIR, "run_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(run_summary, f, indent=2)
    
    print(f"\n📊 Run summary saved to: {summary_file}")
    
    # 收集结果
    if successful > 0:
        collect_results()
    
    print("\n" + "="*70)
    print("🎉 Batch training completed!")
    print("="*70)

if __name__ == "__main__":
    main()


