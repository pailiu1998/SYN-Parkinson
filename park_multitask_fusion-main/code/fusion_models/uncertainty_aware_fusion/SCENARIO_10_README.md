# Scenario 10: Add Synthetic to Train Data with New Rows

## 📋 概述

**场景 10** 是在训练数据中添加合成数据（以新行的方式添加），用于测试数据增强对融合模型性能的影响。

### 数据配置

- **数据位置**: `/localdisk2/pliu/park_multitask_fusion-main/data/aligned_synthetic_format/add_synthetic_to_train_data_with_new_rows/`
- **数据特点**: 在原始训练数据基础上，添加合成样本作为新的行（增加训练集大小）

### 文件列表

```
add_synthetic_to_train_data_with_new_rows/
├── features_demography_diagnosis.csv  (5.2 MB)   # Finger tapping features
├── wavlm_fox_features.csv            (23 MB)     # Audio (Quick) features
├── facial_dataset.csv                (882 KB)    # Facial expression features
├── test_set_participants.txt         (4.1 KB)    # Test set patient IDs
├── dev_set_participants.txt          (4.2 KB)    # Dev set patient IDs
└── all_task_ids.txt                  (13 KB)     # All participant IDs
```

---

## 🎯 与其他场景的区别

| Scenario | 数据组成 | 合成数据方式 |
|----------|---------|------------|
| **Scenario 1-9** | 混合/单一来源 | 替换原有数据 |
| **Scenario 10** | 真实数据 + 合成数据 | **添加新行（数据增强）** |

**关键区别**：
- Scenarios 1-9: 用合成数据**替换**部分真实数据
- Scenario 10: 在真实数据基础上**添加**合成数据（扩充训练集）

---

## 📁 已创建的文件

### 1. `constants_scenario_10.py`

配置文件，定义数据路径和模型参数：

```python
import os

BASE_DIR = os.getcwd()+"/../../../"

# 场景10: 在训练数据中添加合成数据（新行方式）
FINGER_FEATURES_FILE = os.path.join(BASE_DIR,"data/aligned_synthetic_format/add_synthetic_to_train_data_with_new_rows/features_demography_diagnosis.csv")
AUDIO_FEATURES_FILE = os.path.join(BASE_DIR,"data/aligned_synthetic_format/add_synthetic_to_train_data_with_new_rows/wavlm_fox_features.csv")
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/aligned_synthetic_format/add_synthetic_to_train_data_with_new_rows/facial_dataset.csv")

MODEL_BASE_PATH = os.path.join(BASE_DIR,"models")

FACIAL_EXPRESSIONS = {
    'smile': True,
    'surprise': False,
    'disgust': False
}

MODEL_SUBSETS = {
    0: ['finger_model_both_hand_fusion_baal', 'fox_model_best_auroc_baal', 'facial_expression_smile_best_auroc_baal'],
}
```

### 2. `uncertainty_aware_fusion_scenario_10.py`

训练脚本，使用 scenario 10 的配置：

```python
from constants_scenario_10 import *
# ... (其余代码与其他 scenario 相同)
```

### 3. `run_scenario_10_test.sh`

快速测试脚本，用于验证配置是否正确。

---

## 🚀 使用方法

### 方法 1: 快速测试（推荐先运行）

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

# 给脚本添加执行权限
chmod +x run_scenario_10_test.sh

# 运行快速测试（5 epochs）
./run_scenario_10_test.sh
```

### 方法 2: 单次完整训练

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

# 使用默认参数训练（244 epochs）
python uncertainty_aware_fusion_scenario_10.py --seed 42

# 或指定更多参数
python uncertainty_aware_fusion_scenario_10.py \
  --seed 42 \
  --num_epochs 244 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --dropout_prob 0.25
```

### 方法 3: 后台训练

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

# 后台运行，输出到日志文件
nohup python uncertainty_aware_fusion_scenario_10.py \
  --seed 42 \
  --num_epochs 244 \
  > scenario_10_training.log 2>&1 &

# 查看进程
ps aux | grep scenario_10 | grep python

# 监控训练进度
tail -f scenario_10_training.log
```

### 方法 4: 批量训练（多个 seeds）

创建批量训练脚本 `run_scenario_10_multiple_seeds.sh`:

```bash
#!/bin/bash

for seed in {0..10}
do
  echo "Training with seed $seed"
  python uncertainty_aware_fusion_scenario_10.py \
    --seed $seed \
    --num_epochs 244 \
    > scenario_10_seed_${seed}.log 2>&1
done
```

---

## 🎛️ 主要超参数

基于其他 scenarios 的最佳配置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--learning_rate` | 0.001 | 学习率 |
| `--dropout_prob` | 0.25 | Dropout 概率 |
| `--num_epochs` | 244 | 训练轮数 |
| `--batch_size` | 64 | 批次大小 |
| `--hidden_dim` | 128 | 隐藏层维度 |
| `--query_dim` | 64 | Query 向量维度 |
| `--last_hidden_dim` | 8 | 最后隐藏层维度 |
| `--uncertainty_weight` | 0.01 | 不确定性权重 |
| `--num_trials` | 30 | MC Dropout 采样次数 |
| `--optimizer` | AdamW | 优化器 |
| `--use_scheduler` | yes | 使用学习率调度器 |
| `--scheduler` | reduce | 调度器类型 |
| `--patience` | 6 | 早停耐心值 |

---

## 📊 预期效果

### 研究问题

Scenario 10 旨在回答：

1. **数据增强的效果**
   - 在训练集中添加合成数据能否提升模型性能？
   - 相比替换方式（Scenarios 1-9），添加方式是否更有效？

2. **训练集大小的影响**
   - 更大的训练集是否总是带来更好的性能？
   - 合成数据的质量如何影响增强效果？

3. **泛化能力**
   - 使用增强数据训练的模型在真实测试集上表现如何？
   - 是否会过拟合到合成数据的分布？

### 对比实验建议

建议对比以下场景的结果：

| 对比组 | 说明 |
|--------|------|
| **Scenario 10 vs 原始数据** | 评估数据增强的整体效果 |
| **Scenario 10 vs Scenarios 4-6** | 对比"添加"vs"替换"两种合成数据使用方式 |
| **不同增强比例** | 如果数据允许，测试不同的合成/真实数据比例 |

---

## 📈 评估指标

训练过程会输出以下指标：

### Dev Set 指标（每个 epoch）
- Dev F1 Score
- Dev AUROC
- Dev Balanced Accuracy
- Dev Loss

### Test Set 指标（最终）
- Test AUROC
- Test F1 Score
- Test Balanced Accuracy
- Test Accuracy
- Test Precision
- Test Recall
- Test ECE (Expected Calibration Error)
- Test Brier Score
- Test Coverage

所有指标会自动记录到 wandb。

---

## 🔍 数据验证

在训练前，建议验证数据：

```python
import pandas as pd

# 检查数据大小
base_path = "/localdisk2/pliu/park_multitask_fusion-main/data/aligned_synthetic_format"

# Scenario 10 数据
df_s10_finger = pd.read_csv(f"{base_path}/add_synthetic_to_train_data_with_new_rows/features_demography_diagnosis.csv")
df_s10_audio = pd.read_csv(f"{base_path}/add_synthetic_to_train_data_with_new_rows/wavlm_fox_features.csv")
df_s10_facial = pd.read_csv(f"{base_path}/add_synthetic_to_train_data_with_new_rows/facial_dataset.csv")

print(f"Scenario 10 - Finger: {len(df_s10_finger)} rows")
print(f"Scenario 10 - Audio:  {len(df_s10_audio)} rows")
print(f"Scenario 10 - Facial: {len(df_s10_facial)} rows")

# 对比原始数据（如果需要）
df_orig_finger = pd.read_csv(f"{base_path}/../finger_tapping/features_demography_diagnosis_Nov22_2023.csv")
df_orig_audio = pd.read_csv(f"{base_path}/../quick_brown_fox/wavlm_fox_features.csv")
df_orig_facial = pd.read_csv(f"{base_path}/../facial_expression_smile/facial_dataset.csv")

print(f"\nOriginal - Finger: {len(df_orig_finger)} rows")
print(f"Original - Audio:  {len(df_orig_audio)} rows")
print(f"Original - Facial: {len(df_orig_facial)} rows")

print(f"\n增加的样本数:")
print(f"Finger: +{len(df_s10_finger) - len(df_orig_finger)} rows")
print(f"Audio:  +{len(df_s10_audio) - len(df_orig_audio)} rows")
print(f"Facial: +{len(df_s10_facial) - len(df_orig_facial)} rows")
```

---

## 💡 最佳实践

1. **先运行快速测试**
   ```bash
   ./run_scenario_10_test.sh
   ```
   验证数据加载和模型构建无误。

2. **监控训练指标**
   - 实时查看 `Dev Loss` 和 `Dev AUROC`
   - 在 wandb 中对比不同 seeds 的结果

3. **多 seed 评估**
   - 运行至少 10-20 个不同的 seeds
   - 计算性能指标的均值和置信区间
   - 与其他 scenarios 进行统计显著性检验

4. **保存结果**
   - 每个 seed 的结果会自动保存为 JSON 文件
   - `fusion_model_results_test.json`
   - `fusion_model_results_dev.json`

---

## 🆘 故障排查

### 问题 1: 数据加载错误

**错误信息**:
```
FileNotFoundError: [Errno 2] No such file or directory: '...csv'
```

**解决方法**:
1. 检查数据文件是否存在：
   ```bash
   ls -lh /localdisk2/pliu/park_multitask_fusion-main/data/aligned_synthetic_format/add_synthetic_to_train_data_with_new_rows/
   ```

2. 验证 `constants_scenario_10.py` 中的路径是否正确。

### 问题 2: GPU 内存不足

**解决方法**:
```bash
# 减小 batch_size
python uncertainty_aware_fusion_scenario_10.py --seed 42 --batch_size 32

# 或减少 num_trials
python uncertainty_aware_fusion_scenario_10.py --seed 42 --num_trials 15
```

### 问题 3: 模型不收敛

**解决方法**:
1. 调整学习率：
   ```bash
   python uncertainty_aware_fusion_scenario_10.py --seed 42 --learning_rate 0.0001
   ```

2. 增加训练轮数：
   ```bash
   python uncertainty_aware_fusion_scenario_10.py --seed 42 --num_epochs 300
   ```

---

## 📚 相关文档

- [融合模型主文档](./README.md) *(如果存在)*
- [单模态训练指南](../../unimodal_models/UNIMODAL_TRAINING_GUIDE.md)
- [Scenarios 4-6 对比分析](./SCENARIOS_4_5_6_COMPARISON.md)

---

## 🎉 开始训练！

```bash
# 1. 快速测试
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion
chmod +x run_scenario_10_test.sh
./run_scenario_10_test.sh

# 2. 如果测试成功，开始完整训练
nohup python uncertainty_aware_fusion_scenario_10.py --seed 42 > scenario_10_training.log 2>&1 &

# 3. 监控训练
tail -f scenario_10_training.log
```

---

**创建时间**: 2024-12-22  
**场景类型**: Data Augmentation (训练数据增强)  
**状态**: ✅ Ready to use

