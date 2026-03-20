# 🚀 Scenarios 11-13 快速启动指南

## ⚡ 一键测试（单个 seed）

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

# 测试 Scenario 11 (Smile synthetic test)
python uncertainty_aware_fusion_scenario_11.py --seed 0 --num_epochs 5

# 测试 Scenario 12 (Finger synthetic test)
python uncertainty_aware_fusion_scenario_12.py --seed 0 --num_epochs 5

# 测试 Scenario 13 (Quick synthetic test)
python uncertainty_aware_fusion_scenario_13.py --seed 0 --num_epochs 5
```

---

## 🎯 标准训练（单个 seed）

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

# Scenario 11
python uncertainty_aware_fusion_scenario_11.py \
  --seed 42 \
  --num_epochs 244 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --dropout_prob 0.25

# Scenario 12
python uncertainty_aware_fusion_scenario_12.py \
  --seed 42 \
  --num_epochs 244 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --dropout_prob 0.25

# Scenario 13
python uncertainty_aware_fusion_scenario_13.py \
  --seed 42 \
  --num_epochs 244 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --dropout_prob 0.25
```

---

## 🔄 后台运行（三个场景同时）

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

# 同时启动三个场景
nohup python uncertainty_aware_fusion_scenario_11.py --seed 42 --num_epochs 244 > scenario_11.log 2>&1 &
nohup python uncertainty_aware_fusion_scenario_12.py --seed 42 --num_epochs 244 > scenario_12.log 2>&1 &
nohup python uncertainty_aware_fusion_scenario_13.py --seed 42 --num_epochs 244 > scenario_13.log 2>&1 &

echo "✅ 三个场景已启动！"
echo "📊 监控命令:"
echo "  tail -f scenario_11.log"
echo "  tail -f scenario_12.log"
echo "  tail -f scenario_13.log"
```

---

## 📊 监控训练进度

### 查看实时日志
```bash
# Scenario 11
tail -f scenario_11.log

# Scenario 12
tail -f scenario_12.log

# Scenario 13
tail -f scenario_13.log
```

### 查看 Dev 指标
```bash
# 只看 Dev F1, AUROC, Balanced Accuracy
tail -f scenario_11.log | grep "Dev F1"

# 查看所有三个场景的进度
watch -n 10 'tail -3 scenario_11.log; echo "---"; tail -3 scenario_12.log; echo "---"; tail -3 scenario_13.log'
```

### 检查运行进程
```bash
ps aux | grep "uncertainty_aware_fusion_scenario_1[1-3]" | grep python
```

---

## 🛑 停止训练

```bash
# 停止所有三个场景
pkill -f "uncertainty_aware_fusion_scenario_11.py"
pkill -f "uncertainty_aware_fusion_scenario_12.py"
pkill -f "uncertainty_aware_fusion_scenario_13.py"

# 或者一次性停止
pkill -f "uncertainty_aware_fusion_scenario_1[1-3].py"
```

---

## 📈 训练输出示例

每个 epoch 会打印：
```
Epoch 0: Dev F1: 0.7234, Dev AUROC: 0.8456, Dev Balanced Accuracy: 0.7891, Dev Loss: 0.4532
Epoch 1: Dev F1: 0.7456, Dev AUROC: 0.8567, Dev Balanced Accuracy: 0.8012, Dev Loss: 0.4321
...
```

训练结束后：
```
======================================================================
BEST DEV SET METRICS
======================================================================
Dev F1 Score:           0.7823
Dev AUROC:              0.8945
Dev Balanced Accuracy:  0.8456
Dev Accuracy:           0.8523
Dev Loss:               0.3876
Dev ECE:                0.0234
======================================================================

TEST SET METRICS
======================================================================
Test F1 Score:          0.7956
Test AUROC:             0.9012
Test Balanced Accuracy: 0.8567
...
```

---

## 🔬 对比分析

运行完所有三个场景后，对比它们的性能：

```bash
# 假设都已完成，查看各自的最佳 dev 指标
grep "BEST DEV SET METRICS" -A 10 scenario_11.log
grep "BEST DEV SET METRICS" -A 10 scenario_12.log
grep "BEST DEV SET METRICS" -A 10 scenario_13.log

# 查看 test 指标
grep "TEST SET METRICS" -A 10 scenario_11.log
grep "TEST SET METRICS" -A 10 scenario_12.log
grep "TEST SET METRICS" -A 10 scenario_13.log
```

---

## 📊 预期对比结果

| Scenario | Synthetic Test 模态 | 替换率 | 预期影响 |
|----------|-------------------|--------|---------|
| 11 | Smile | 62.3% | 中等性能下降 |
| 12 | Finger | 88.4% | ⚠️ 较大性能下降（替换率最高）|
| 13 | Quick | 65.2% | 中等性能下降 |

**假设**: Scenario 12 可能表现最差，因为 Finger 测试集有 88.4% 是合成数据。

---

## ⏱️ 预计时间

- **单个 epoch**: ~30-60 秒
- **244 epochs**: ~2-4 小时
- **三个场景同时运行**: ~2-4 小时（如果有足够 GPU）

---

## 📁 输出文件

每个场景会生成：
```
fusion_model_results_test.json    # Test set 性能指标
fusion_model_results_dev.json     # Dev set 性能指标
best_fusion_model.pth              # 最佳模型权重
```

---

## ✅ 验证配置

运行前验证配置是否正确：

```bash
# 检查数据路径
python -c "from constants_scenario_11 import *; print(FINGER_FEATURES_FILE)"
python -c "from constants_scenario_12 import *; print(FINGER_FEATURES_FILE)"
python -c "from constants_scenario_13 import *; print(FINGER_FEATURES_FILE)"

# 预期输出应包含正确的路径:
# .../synthetic_train_real_finger_quick/...
# .../synthetic_train_real_quick_smile/...
# .../synthetic_train_real_smile_finger/...
```

---

## 🎉 开始吧！

选择一个命令开始训练：

```bash
# 快速测试（5 epochs，约 5 分钟）
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion
python uncertainty_aware_fusion_scenario_11.py --seed 0 --num_epochs 5

# 完整训练（244 epochs，约 2-4 小时，后台运行）
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion
nohup python uncertainty_aware_fusion_scenario_11.py --seed 42 --num_epochs 244 > scenario_11.log 2>&1 &
tail -f scenario_11.log
```

祝训练顺利！🚀

