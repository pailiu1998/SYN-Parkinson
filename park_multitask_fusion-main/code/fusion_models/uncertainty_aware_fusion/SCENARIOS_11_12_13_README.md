# Scenarios 11, 12, 13 - Synthetic Train with Selective Real Test

## 📋 场景概述

这三个场景都采用了**统一的训练集增强策略**，但在**测试集**上有所不同。核心设计是：
- **训练集**：所有模态都添加了合成数据（+647个样本）
- **测试集**：每个场景只有一个模态使用合成数据，其他两个模态使用真实数据

---

## 🎯 实验目的

评估**单个模态在测试集使用合成数据**对融合模型整体性能的影响：
1. 哪个模态的合成数据质量最好/最差？
2. 模型对不同模态合成数据的敏感性如何？
3. 训练集增强 + 测试集合成数据的组合效果如何？

---

## 📊 三个场景配置

### Scenario 11: `synthetic_train_real_finger_quick`

**目标**: 评估 **Smile synthetic test** 的影响

| 模态 | 训练集 | 测试集 | 说明 |
|------|--------|--------|------|
| **Finger** | ✅ 有合成增强 | ✅ 真实数据 | 保持原始测试 |
| **Quick** | ✅ 有合成增强 | ✅ 真实数据 | 保持原始测试 |
| **Smile** | ✅ 有合成增强 | ⚠️ **合成数据** | **62.3% 替换为合成** |

**文件位置**:
- `constants_scenario_11.py`
- `uncertainty_aware_fusion_scenario_11.py`
- 数据: `data/aligned_synthetic_format/synthetic_train_real_finger_quick/`

---

### Scenario 12: `synthetic_train_real_quick_smile`

**目标**: 评估 **Finger synthetic test** 的影响

| 模态 | 训练集 | 测试集 | 说明 |
|------|--------|--------|------|
| **Finger** | ✅ 有合成增强 | ⚠️ **合成数据** | **88.4% 替换为合成** |
| **Quick** | ✅ 有合成增强 | ✅ 真实数据 | 保持原始测试 |
| **Smile** | ✅ 有合成增强 | ✅ 真实数据 | 保持原始测试 |

**文件位置**:
- `constants_scenario_12.py`
- `uncertainty_aware_fusion_scenario_12.py`
- 数据: `data/aligned_synthetic_format/synthetic_train_real_quick_smile/`

---

### Scenario 13: `synthetic_train_real_smile_finger`

**目标**: 评估 **Quick synthetic test** 的影响

| 模态 | 训练集 | 测试集 | 说明 |
|------|--------|--------|------|
| **Finger** | ✅ 有合成增强 | ✅ 真实数据 | 保持原始测试 |
| **Quick** | ✅ 有合成增强 | ⚠️ **合成数据** | **65.2% 替换为合成** |
| **Smile** | ✅ 有合成增强 | ✅ 真实数据 | 保持原始测试 |

**文件位置**:
- `constants_scenario_13.py`
- `uncertainty_aware_fusion_scenario_13.py`
- 数据: `data/aligned_synthetic_format/synthetic_train_real_smile_finger/`

---

## 📈 数据统计

### 训练集合成增强（所有场景相同）
```
- Smile:  +4 个样本
- Quick:  +25 个样本
- Finger: +618 个样本（左右手各 309）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总计:    +647 个样本
```

### 测试集合成替换率
```
- Smile (Scenario 11):  200/321 = 62.3%
- Finger (Scenario 12): 206/233 = 88.4% ⚠️ 最高
- Quick (Scenario 13):  202/310 = 65.2%
```

> ⚠️ **注意**: Scenario 12 的 Finger 测试集有 88.4% 是合成数据，这是最高的替换率！

---

## 🚀 使用方法

### 单独运行某个场景

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

# Scenario 11 (Smile synthetic test)
python uncertainty_aware_fusion_scenario_11.py --seed 42 --num_epochs 244

# Scenario 12 (Finger synthetic test)
python uncertainty_aware_fusion_scenario_12.py --seed 42 --num_epochs 244

# Scenario 13 (Quick synthetic test)
python uncertainty_aware_fusion_scenario_13.py --seed 42 --num_epochs 244
```

### 后台运行

```bash
# Scenario 11
nohup python uncertainty_aware_fusion_scenario_11.py \
  --seed 42 --num_epochs 244 \
  > scenario_11.log 2>&1 &

# Scenario 12
nohup python uncertainty_aware_fusion_scenario_12.py \
  --seed 42 --num_epochs 244 \
  > scenario_12.log 2>&1 &

# Scenario 13
nohup python uncertainty_aware_fusion_scenario_13.py \
  --seed 42 --num_epochs 244 \
  > scenario_13.log 2>&1 &
```

### 监控训练

```bash
# 查看日志
tail -f scenario_11.log

# 查看 Dev 指标（实时打印）
tail -f scenario_11.log | grep "Dev F1"
```

---

## 🔬 与其他场景的对比

| Scenario | 训练集策略 | 测试集策略 | 研究重点 |
|----------|----------|----------|---------|
| **1-3** | 原始数据 | 部分合成替换 | 评估部分模态合成的影响 |
| **4-6** | 混合数据 | 部分模态合成 | 评估混合训练的效果 |
| **7-9** | 单模态合成 | 双模态合成 | 评估单到双模态的迁移 |
| **10** | 所有模态增强 | 全真实 | 评估数据增强的效果 |
| **11-13** | ⭐ 所有模态增强 | ⭐ 单模态合成 | ⭐ **评估单模态测试合成的影响** |

---

## 📊 预期分析

运行完这三个场景后，可以回答：

1. **哪个模态的合成数据质量最好？**
   - 比较 Scenario 11, 12, 13 的测试集性能
   - 性能下降最小的模态 = 合成数据质量最好

2. **模型对哪个模态最敏感？**
   - 性能下降最大的场景 = 模型对该模态最依赖

3. **训练增强 + 测试合成的组合效果**
   - 对比 Scenario 10 (全真实测试) vs Scenarios 11-13
   - 评估测试集合成数据的影响程度

---

## 💡 建议的实验流程

```bash
# 1. 先运行 Scenario 10 作为基线（全真实测试）
python uncertainty_aware_fusion_scenario_10.py --seed 42

# 2. 运行 Scenarios 11-13（单模态测试合成）
python uncertainty_aware_fusion_scenario_11.py --seed 42
python uncertainty_aware_fusion_scenario_12.py --seed 42
python uncertainty_aware_fusion_scenario_13.py --seed 42

# 3. 对比分析
# 计算每个场景相对于 Scenario 10 的性能下降
# 性能下降 = (Scenario 10 指标) - (Scenario X 指标)
```

---

## 📁 文件清单

**配置文件** (3个):
- `constants_scenario_11.py` (1.3 KB)
- `constants_scenario_12.py` (1.3 KB)
- `constants_scenario_13.py` (1.3 KB)

**训练脚本** (3个):
- `uncertainty_aware_fusion_scenario_11.py` (60 KB)
- `uncertainty_aware_fusion_scenario_12.py` (60 KB)
- `uncertainty_aware_fusion_scenario_13.py` (60 KB)

**数据目录** (3个):
- `data/aligned_synthetic_format/synthetic_train_real_finger_quick/`
- `data/aligned_synthetic_format/synthetic_train_real_quick_smile/`
- `data/aligned_synthetic_format/synthetic_train_real_smile_finger/`

---

## ✅ 验证

验证文件是否正确创建：

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

# 检查文件存在
ls -lh constants_scenario_1[1-3].py
ls -lh uncertainty_aware_fusion_scenario_1[1-3].py

# 验证 import 语句
grep "from constants_scenario" uncertainty_aware_fusion_scenario_1[1-3].py

# 预期输出:
# uncertainty_aware_fusion_scenario_11.py:from constants_scenario_11 import *
# uncertainty_aware_fusion_scenario_12.py:from constants_scenario_12 import *
# uncertainty_aware_fusion_scenario_13.py:from constants_scenario_13 import *
```

---

## 🎉 已完成

✅ 所有 Scenario 11, 12, 13 文件已成功创建并配置完成！

如需批量运行多个 seeds 或进行统计分析，可参考 `run_scenario_10_sequential.py` 和 `analyze_100seeds_results.py` 创建类似的批处理脚本。

