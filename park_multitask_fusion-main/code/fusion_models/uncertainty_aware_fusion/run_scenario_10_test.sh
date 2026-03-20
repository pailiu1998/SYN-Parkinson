#!/bin/bash

# ============================================================================
# Scenario 10 快速测试脚本
# 场景10: 在训练数据中添加合成数据（新行方式）
# ============================================================================

echo "=========================================="
echo "Scenario 10 测试运行"
echo "数据源: add_synthetic_to_train_data_with_new_rows"
echo "=========================================="

cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

# 快速测试（5 epochs）
echo "开始快速测试 (5 epochs)..."
python uncertainty_aware_fusion_scenario_10.py \
  --seed 0 \
  --num_epochs 5 \
  --batch_size 64

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "如果测试成功，可以使用以下命令进行完整训练："
echo ""
echo "# 单次训练（244 epochs）"
echo "python uncertainty_aware_fusion_scenario_10.py --seed 42 --num_epochs 244"
echo ""
echo "# 后台训练"
echo "nohup python uncertainty_aware_fusion_scenario_10.py --seed 42 --num_epochs 244 > scenario_10_training.log 2>&1 &"
echo ""

