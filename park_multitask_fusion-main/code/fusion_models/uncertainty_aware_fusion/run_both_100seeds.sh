#!/bin/bash

# ============================================================================
# 同时运行 Scenario 10 和 Original Fusion 的 100 seeds 批量训练
# 每个并行运行10个进程
# ============================================================================

cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

echo "=========================================="
echo "Starting batch training for both models"
echo "=========================================="
echo "Time: $(date)"
echo ""
echo "1. Scenario 10 (add_synthetic_to_train_data_with_new_rows)"
echo "   - 100 seeds (0-99)"
echo "   - 10 parallel jobs"
echo "   - Output: scenario_10_results/"
echo ""
echo "2. Original Fusion (baseline)"
echo "   - 100 seeds (0-99)"
echo "   - 10 parallel jobs"
echo "   - Output: original_fusion_results/"
echo ""
echo "=========================================="
echo ""

# 运行 Scenario 10 (后台)
echo "Starting Scenario 10 batch training..."
nohup python run_scenario_10_100seeds.py > scenario_10_batch_log.txt 2>&1 &
PID_SCENARIO_10=$!
echo "Scenario 10 started with PID: $PID_SCENARIO_10"

# 等待几秒，避免GPU冲突
sleep 5

# 运行 Original Fusion (后台)
echo "Starting Original Fusion batch training..."
nohup python run_original_fusion_100seeds.py > original_fusion_batch_log.txt 2>&1 &
PID_ORIGINAL=$!
echo "Original Fusion started with PID: $PID_ORIGINAL"

echo ""
echo "=========================================="
echo "Both batch trainings started!"
echo "=========================================="
echo ""
echo "📊 Monitor progress:"
echo ""
echo "# Scenario 10"
echo "tail -f scenario_10_batch_log.txt"
echo "ps -p $PID_SCENARIO_10"
echo ""
echo "# Original Fusion"
echo "tail -f original_fusion_batch_log.txt"
echo "ps -p $PID_ORIGINAL"
echo ""
echo "=========================================="
echo "📁 Results will be saved to:"
echo ""
echo "Scenario 10:"
echo "  - scenario_10_results/"
echo "  - scenario_10_results/all_seeds_results.json"
echo "  - scenario_10_results/run_summary.json"
echo ""
echo "Original Fusion:"
echo "  - original_fusion_results/"
echo "  - original_fusion_results/all_seeds_results.json"
echo "  - original_fusion_results/run_summary.json"
echo ""
echo "=========================================="
echo "🔍 Check running processes:"
echo "ps aux | grep 'run_scenario_10_100seeds\\|run_original_fusion_100seeds' | grep python"
echo ""
echo "⏹️  Stop all trainings (if needed):"
echo "kill $PID_SCENARIO_10 $PID_ORIGINAL"
echo ""
echo "=========================================="


