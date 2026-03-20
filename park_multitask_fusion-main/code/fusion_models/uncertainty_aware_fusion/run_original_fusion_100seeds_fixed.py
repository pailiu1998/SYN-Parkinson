#!/usr/bin/env python3
"""
Original Fusion Model 批量训练脚本 - 100 seeds (已修复文件竞争问题)
并行运行：每次10个进程
"""

import subprocess
import os
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# 配置
SCRIPT_NAME = "uncertainty_aware_fusion.py"
NUM_SEEDS = 100
START_SEED = 0
PARALLEL_JOBS = 10
OUTPUT_DIR = "original_fusion_results"
TIMEOUT = 7200  # 2小时超时

def run_experiment(seed):
    """运行单个实验"""
    print(f"[Seed {seed}] Starting...")
    
    # 创建输出目录
    seed_dir = os.path.join(OUTPUT_DIR, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    
    # 获取训练脚本的绝对路径
    script_path = os.path.abspath(SCRIPT_NAME)
    
    # 构建命令 - 使用绝对路径
    cmd = [
        "python", script_path,
        "--seed", str(seed),
        "--num_epochs", "244",
        "--batch_size", "64",
        "--learning_rate", "0.001",
        "--dropout_prob", "0.25"
    ]
    
    start_time = time.time()
    
    try:
        # 在seed专用目录中运行训练 ⭐ 使用绝对路径
        result = subprocess.run(
            cmd,
            text=True,
            timeout=TIMEOUT,
            cwd=seed_dir  # ⭐ 每个seed在自己的目录中运行
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            # 检查结果文件是否生成
            test_file = os.path.join(seed_dir, "fusion_model_results_test.json")
            dev_file = os.path.join(seed_dir, "fusion_model_results_dev.json")
            
            if os.path.exists(test_file) and os.path.exists(dev_file):
                print(f"[Seed {seed}] ✅ Completed in {elapsed_time:.1f}s")
                return {"seed": seed, "status": "success", "time": elapsed_time}
            else:
                print(f"[Seed {seed}] ⚠️ Completed but missing result files")
                return {"seed": seed, "status": "incomplete", "time": elapsed_time, 
                       "error": "Missing result files"}
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
        print(f"   Failed seeds: {failed_seeds[:10]}{'...' if len(failed_seeds) > 10 else ''}")
    
    print(f"\n📁 Results saved to: {summary_file}")
    
    return results

def main():
    """主函数"""
    print("="*70)
    print("Original Fusion Model - Batch Training with 100 Seeds (Fixed Version)")
    print("="*70)
    print(f"Script: {SCRIPT_NAME}")
    print(f"Seeds: {START_SEED} to {START_SEED + NUM_SEEDS - 1}")
    print(f"Parallel jobs: {PARALLEL_JOBS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"⭐ Each seed runs in its own directory to avoid file conflicts")
    print("="*70 + "\n")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 记录开始时间
    total_start_time = time.time()
    
    # 使用进程池并行运行
    seeds = list(range(START_SEED, START_SEED + NUM_SEEDS))
    run_results = []
    
    with ProcessPoolExecutor(max_workers=PARALLEL_JOBS) as executor:
        # 提交所有任务
        future_to_seed = {executor.submit(run_experiment, seed): seed for seed in seeds}
        
        # 收集结果
        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                result = future.result()
                run_results.append(result)
            except Exception as e:
                print(f"[Seed {seed}] Exception: {str(e)}")
                run_results.append({"seed": seed, "status": "exception", "error": str(e)})
    
    # 计算总时间
    total_elapsed_time = time.time() - total_start_time
    
    # 统计运行结果
    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)
    
    successful = sum(1 for r in run_results if r.get("status") == "success")
    incomplete = sum(1 for r in run_results if r.get("status") == "incomplete")
    failed = sum(1 for r in run_results if r.get("status") == "failed")
    timeout = sum(1 for r in run_results if r.get("status") == "timeout")
    errors = sum(1 for r in run_results if r.get("status") in ["error", "exception"])
    
    print(f"Total seeds: {NUM_SEEDS}")
    print(f"✅ Successful: {successful}")
    print(f"⚠️  Incomplete: {incomplete}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Timeout: {timeout}")
    print(f"🔥 Errors: {errors}")
    print(f"⏰ Total time: {total_elapsed_time/3600:.2f} hours")
    
    # 保存运行统计
    run_summary = {
        "total_seeds": NUM_SEEDS,
        "successful": successful,
        "incomplete": incomplete,
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

