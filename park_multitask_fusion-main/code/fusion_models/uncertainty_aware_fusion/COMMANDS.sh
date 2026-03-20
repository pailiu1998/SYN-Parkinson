#!/bin/bash
# 一键命令集合 - 复制粘贴即可使用

# ============================================================================
# 1. 同时运行 Scenario 4 和 5（推荐）
# ============================================================================
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion
./run_scenarios_4_and_5.sh

# ============================================================================
# 2. 监控进度 - 实时查看日志
# ============================================================================
# Scenario 4
tail -f scenario_4_run.log

# Scenario 5
tail -f scenario_5_run.log

# ============================================================================
# 3. 查看进程状态
# ============================================================================
ps aux | grep run_scenario

# ============================================================================
# 4. 查看GPU使用
# ============================================================================
watch -n 1 nvidia-smi

# ============================================================================
# 5. 查看已完成的seeds数量
# ============================================================================
echo "Scenario 4 completed seeds:"
ls scenario_4_results_*/test_seed_*.json 2>/dev/null | wc -l

echo "Scenario 5 completed seeds:"
ls scenario_5_results_*/test_seed_*.json 2>/dev/null | wc -l

# ============================================================================
# 6. 查看实时Dev指标
# ============================================================================
# 最新的10个epoch
grep "Dev F1" scenario_4_run.log | tail -10
grep "Dev F1" scenario_5_run.log | tail -10

# ============================================================================
# 7. 查看最佳Dev指标
# ============================================================================
grep -A 6 "BEST DEV SET METRICS" scenario_4_run.log | tail -20
grep -A 6 "BEST DEV SET METRICS" scenario_5_run.log | tail -20

# ============================================================================
# 8. 停止实验（如果需要）
# ============================================================================
# 查找PID
ps aux | grep run_scenario

# 停止所有
pkill -f "run_scenario_[45]"

# ============================================================================
# 9. 单独运行某个场景
# ============================================================================
# 只运行 Scenario 4
nohup python run_scenario_4_multiple_seeds.py > scenario_4_run.log 2>&1 &

# 只运行 Scenario 5
nohup python run_scenario_5_multiple_seeds.py > scenario_5_run.log 2>&1 &

# 只运行 Scenario 6
nohup python run_scenario_6_multiple_seeds.py > scenario_6_run.log 2>&1 &

# ============================================================================
# 10. 使用不同GPU（如果有多个GPU）
# ============================================================================
# Scenario 4 使用 GPU 0
CUDA_VISIBLE_DEVICES=0 nohup python run_scenario_4_multiple_seeds.py > scenario_4_run.log 2>&1 &

# Scenario 5 使用 GPU 1
CUDA_VISIBLE_DEVICES=1 nohup python run_scenario_5_multiple_seeds.py > scenario_5_run.log 2>&1 &

# ============================================================================
# 11. 查看帮助文档
# ============================================================================
cat START_SCENARIOS_4_5.txt
cat RUN_SCENARIOS_4_5_GUIDE.md
cat QUICKSTART.md


