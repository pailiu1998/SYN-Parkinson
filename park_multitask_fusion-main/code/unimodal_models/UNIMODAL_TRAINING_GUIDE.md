# 单模态模型训练指南

## 📋 目录
1. [概述](#概述)
2. [模型架构](#模型架构)
3. [数据路径配置](#数据路径配置)
4. [训练单个模态](#训练单个模态)
5. [超参数说明](#超参数说明)
6. [批量训练（使用 wandb sweep）](#批量训练)
7. [模型评估指标](#模型评估指标)
8. [常见问题](#常见问题)

---

## 概述

本代码库支持三种单模态模型的训练：

| 模态 | 数据类型 | 脚本位置 |
|------|----------|----------|
| **Finger Tapping** (手指敲击) | 运动特征 | `finger_tapping/unimodal_finger_baal.py` |
| **Quick Brown Fox** (语音) | 音频特征 (WavLM) | `quick_brown_fox/unimodal_fox_baal.py` |
| **Facial Expression** (面部表情) | 面部特征 | `facial_expression_smile/unimodal_smile_baal.py` |

每个模态都使用 **Bayesian Active Learning (BAAL)** 框架进行不确定性估计。

---

## 模型架构

### 1. ShallowANN (推荐)
简单的单层神经网络，适合特征已经比较好的情况：

```python
class ShallowANN(nn.Module):
    Input (n_features) 
      ↓
    Linear Layer (n_features → 1)
      ↓
    MC Dropout
      ↓
    Sigmoid
      ↓
    Output (probability)
```

**特点**：
- 参数少，训练快
- 适合高维特征（如 WavLM embeddings）
- 防止过拟合

### 2. ANN (两层网络)
更深的网络，适合需要更多特征转换的情况：

```python
class ANN(nn.Module):
    Input (n_features)
      ↓
    Linear Layer (n_features → n_features/2)
      ↓
    ReLU + MC Dropout
      ↓
    Linear Layer (n_features/2 → 1)
      ↓
    MC Dropout
      ↓
    Sigmoid
      ↓
    Output (probability)
```

---

## 数据路径配置

### Finger Tapping
编辑 `finger_tapping/constants_baal.py`:

```python
import os

BASE_DIR = os.getcwd()+"/../../../"

# 选择数据源（取消注释你想用的）
# 1. 真实数据
# FEATURES_FILE = os.path.join(BASE_DIR,"data/finger_tapping/features_demography_diagnosis_Nov22_2023.csv")

# 2. 合成数据
FEATURES_FILE = os.path.join(BASE_DIR,"data/synthetic_data/merged/finger_embeddings.csv")

# 3. 混合数据 (scenario 特定)
# FEATURES_FILE = os.path.join(BASE_DIR,"data/aligned_synthetic_format/mixed_scenario_4_real_finger_quick/features_demography_diagnosis.csv")

MODEL_TAG = "both_hand_fusion_baal"
MODEL_BASE_PATH = os.path.join(BASE_DIR,f"models/finger_model_{MODEL_TAG}")
```

### Quick Brown Fox (Audio)
编辑 `quick_brown_fox/constants_baal.py`:

```python
import os

BASE_PATH = os.getcwd()+"/../../../"
BASE_DIR = BASE_PATH

# WavLM 特征文件
# WAVLM_FEATURES_FILE = os.path.join(BASE_PATH,"data/quick_brown_fox/wavlm_fox_features.csv")
WAVLM_FEATURES_FILE = os.path.join(BASE_PATH,"data/synthetic_data/merged/quick_embeddings.csv")

MODEL_TAG = "best_auroc_baal"
MODEL_BASE_PATH = os.path.join(BASE_PATH, f"models/fox_model_{MODEL_TAG}")
```

### Facial Expression (Smile)
编辑 `facial_expression_smile/constants_baal.py`:

```python
import os

BASE_PATH = os.getcwd()+"/../../../"
BASE_DIR = BASE_PATH

FACIAL_EXPRESSIONS = {
    'smile': True,      # 使用 smile 表情
    'surprise': False,  # 不使用 surprise
    'disgust': False    # 不使用 disgust
}

# FEATURES_FILE = os.path.join(BASE_PATH,"data/facial_expression_smile/facial_dataset.csv")
FEATURES_FILE = os.path.join(BASE_PATH,"data/synthetic_data/merged/smile_embeddings.csv")

MODEL_TAG = "best_auroc_baal"
MODEL_BASE_PATH = os.path.join(BASE_PATH,f"models/facial_expression_smile_{MODEL_TAG}")
```

---

## 训练单个模态

### 基本训练命令

#### 1. Finger Tapping

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/unimodal_models/finger_tapping

python unimodal_finger_baal.py \
  --model ShallowANN \
  --dropout_prob 0.13951215957675367 \
  --num_trials 300 \
  --num_buckets 20 \
  --hand both \
  --learning_rate 0.6682837019078968 \
  --random_state 526 \
  --seed 526 \
  --use_feature_scaling yes \
  --scaling_method StandardScaler \
  --minority_oversample no \
  --batch_size 512 \
  --num_epochs 73 \
  --drop_correlated no \
  --corr_thr 0.95 \
  --optimizer SGD \
  --momentum 0.8363833208184809 \
  --use_scheduler yes \
  --scheduler step \
  --step_size 22 \
  --gamma 0.6555323541714391
```

#### 2. Quick Brown Fox (Audio)

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/unimodal_models/quick_brown_fox

python unimodal_fox_baal.py \
  --model ShallowANN \
  --dropout_prob 0.08349938684379829 \
  --num_trials 5000 \
  --num_buckets 20 \
  --learning_rate 0.9258448866412824 \
  --random_state 526 \
  --seed 526 \
  --use_feature_scaling no \
  --scaling_method StandardScaler \
  --minority_oversample yes \
  --batch_size 256 \
  --num_epochs 55 \
  --drop_correlated no \
  --optimizer SGD \
  --momentum 0.49459848722229194 \
  --use_scheduler no
```

#### 3. Facial Expression (Smile)

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/unimodal_models/facial_expression_smile

python unimodal_smile_baal.py \
  --model ShallowANN \
  --dropout_prob 0.1 \
  --num_trials 5000 \
  --num_buckets 20 \
  --learning_rate 0.5 \
  --random_state 526 \
  --seed 526 \
  --use_feature_scaling yes \
  --scaling_method StandardScaler \
  --minority_oversample no \
  --batch_size 256 \
  --num_epochs 50 \
  --drop_correlated yes \
  --corr_thr 0.85 \
  --optimizer SGD \
  --momentum 0.9 \
  --use_scheduler yes \
  --scheduler reduce \
  --patience 10 \
  --gamma 0.5
```

### 使用 nohup 后台运行

```bash
# Finger Tapping
cd /localdisk2/pliu/park_multitask_fusion-main/code/unimodal_models/finger_tapping
nohup python unimodal_finger_baal.py --seed 526 --num_epochs 73 > finger_training.log 2>&1 &

# Quick Brown Fox
cd /localdisk2/pliu/park_multitask_fusion-main/code/unimodal_models/quick_brown_fox
nohup python unimodal_fox_baal.py --seed 526 --num_epochs 55 > fox_training.log 2>&1 &

# Facial Expression
cd /localdisk2/pliu/park_multitask_fusion-main/code/unimodal_models/facial_expression_smile
nohup python unimodal_smile_baal.py --seed 526 --num_epochs 50 > smile_training.log 2>&1 &
```

---

## 超参数说明

### 核心参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--model` | 模型架构 | `ShallowANN` 或 `ANN` |
| `--dropout_prob` | Dropout 概率 | 0.1 - 0.2 |
| `--learning_rate` | 学习率 | 0.3 - 0.9 |
| `--batch_size` | 批次大小 | 256 或 512 |
| `--num_epochs` | 训练轮数 | 50 - 100 |
| `--seed` | 随机种子 | 任意整数（确保可重复性） |

### 数据预处理

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--use_feature_scaling` | 是否标准化特征 | `yes` |
| `--scaling_method` | 标准化方法 | `StandardScaler` 或 `MinMaxScaler` |
| `--minority_oversample` | 是否过采样少数类 (SMOTE) | `no` (数据已平衡时) |
| `--drop_correlated` | 是否删除高度相关的特征 | `no` (特征已预处理时) |
| `--corr_thr` | 相关性阈值 | 0.85 - 0.95 |

### 优化器

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--optimizer` | 优化器类型 | `SGD` 或 `AdamW` |
| `--momentum` | SGD 动量 | 0.8 - 0.95 (仅 SGD) |
| `--weight_decay` | 权重衰减 (L2 正则化) | 0.0001 |
| `--beta1` | AdamW beta1 | 0.9 (仅 AdamW) |
| `--beta2` | AdamW beta2 | 0.999 (仅 AdamW) |

### 学习率调度器

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--use_scheduler` | 是否使用学习率调度器 | `yes` |
| `--scheduler` | 调度器类型 | `step` 或 `reduce` |
| `--step_size` | StepLR 步长 | 15 - 25 (仅 step) |
| `--gamma` | 学习率衰减因子 | 0.5 - 0.8 |
| `--patience` | ReduceLROnPlateau 耐心值 | 10 - 20 (仅 reduce) |

### 不确定性估计 (BAAL)

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--num_trials` | MC Dropout 采样次数 | 300 - 5000 |
| `--num_buckets` | ECE 计算的桶数 | 20 |

### Finger Tapping 特有

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--hand` | 使用哪只手的数据 | `left`, `right`, 或 `both` |

---

## 批量训练

### 使用 wandb sweep 进行超参数搜索

#### 1. 创建 sweep 配置文件

创建 `sweep_config.yaml`:

```yaml
program: unimodal_finger_baal.py
method: bayes
metric:
  name: auroc
  goal: maximize
parameters:
  dropout_prob:
    min: 0.05
    max: 0.3
  learning_rate:
    min: 0.1
    max: 1.0
  num_epochs:
    values: [50, 73, 100]
  batch_size:
    values: [256, 512]
  momentum:
    min: 0.5
    max: 0.95
  scheduler:
    values: ['step', 'reduce']
  step_size:
    min: 15
    max: 30
  gamma:
    min: 0.5
    max: 0.9
```

#### 2. 初始化 sweep

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/unimodal_models/finger_tapping

# 登录 wandb (第一次使用)
wandb login

# 创建 sweep
wandb sweep sweep_config.yaml
```

这会输出一个 sweep ID，例如：`5r5tna5x`

#### 3. 启动 sweep agent

```bash
# 单个 agent
wandb agent pliu29/park_final_experiments/5r5tna5x

# 后台运行
nohup wandb agent pliu29/park_final_experiments/5r5tna5x > sweep_agent.log 2>&1 &

# 多个 agents 并行
nohup wandb agent pliu29/park_final_experiments/5r5tna5x > sweep_agent_1.log 2>&1 &
nohup wandb agent pliu29/park_final_experiments/5r5tna5x > sweep_agent_2.log 2>&1 &
nohup wandb agent pliu29/park_final_experiments/5r5tna5x > sweep_agent_3.log 2>&1 &
```

---

## 模型评估指标

训练过程中会计算以下指标：

### Dev Set (验证集) 指标
在每个 epoch 后评估，用于模型选择：
- **Dev Loss**: 验证集损失 (BCE Loss)
- **Dev Accuracy**: 准确率
- **Dev Balanced Accuracy**: 平衡准确率（处理类别不平衡）
- **Dev AUROC**: ROC 曲线下面积
- **Dev F1 Score**: F1 分数
- **Dev ECE**: 期望校准误差（Expected Calibration Error）

### Test Set (测试集) 指标
使用最佳 dev loss 对应的模型在测试集上评估：
- **accuracy**: 准确率
- **weighted_accuracy**: 平衡准确率
- **auroc**: ROC 曲线下面积
- **f1_score**: F1 分数
- **recall**: 召回率
- **precision**: 精确率
- **average_precision**: 平均精确率
- **brier_score**: Brier 分数
- **ECE**: 期望校准误差
- **loss**: 测试集损失

### Wandb 记录

所有指标都会自动记录到 wandb，可以在网页端查看：
- 训练曲线
- 超参数对比
- 模型性能对比

---

## 常见问题

### Q1: 如何更改训练数据源？

**A**: 编辑对应模态的 `constants_baal.py` 文件，修改 `FEATURES_FILE` 路径。

例如，使用 scenario 4 的数据：

```python
# Finger
FEATURES_FILE = os.path.join(BASE_DIR,"data/aligned_synthetic_format/mixed_scenario_4_real_finger_quick/features_demography_diagnosis.csv")

# Quick
WAVLM_FEATURES_FILE = os.path.join(BASE_PATH,"data/aligned_synthetic_format/mixed_scenario_4_real_finger_quick/wavlm_fox_features.csv")

# Smile
FEATURES_FILE = os.path.join(BASE_PATH,"data/aligned_synthetic_format/mixed_scenario_4_real_finger_quick/facial_dataset.csv")
```

### Q2: 模型保存在哪里？

**A**: 模型保存路径由 `constants_baal.py` 中的 `MODEL_BASE_PATH` 定义：

```
models/
  ├── finger_model_both_hand_fusion_baal/
  │   ├── predictive_model/
  │   │   ├── model.pth         # 模型权重
  │   │   └── model_config.json # 配置文件
  │   └── scaler/
  │       └── scaler.pth         # 特征标准化器
  ├── fox_model_best_auroc_baal/
  └── facial_expression_smile_best_auroc_baal/
```

### Q3: 如何加载已训练的模型？

**A**:

```python
import torch
from unimodal_finger_baal import ShallowANN
import pickle

# 加载配置
with open("models/finger_model_both_hand_fusion_baal/predictive_model/model_config.json") as f:
    config = json.load(f)

# 创建模型
model = ShallowANN(n_features=256, drop_prob=config['dropout_prob'])

# 加载权重
model.load_state_dict(torch.load("models/finger_model_both_hand_fusion_baal/predictive_model/model.pth"))
model.eval()

# 加载标准化器（如果使用）
scaler = pickle.load(open("models/finger_model_both_hand_fusion_baal/scaler/scaler.pth", 'rb'))
```

### Q4: 为什么 num_trials 设置不同？

**A**: `num_trials` 是 MC Dropout 的采样次数，用于不确定性估计：

- **Finger Tapping**: 300 次（特征维度较低，采样快）
- **Quick/Smile**: 5000 次（特征维度高，需要更多采样获得稳定的不确定性估计）

更多采样次数 → 更准确的不确定性估计，但训练时间更长。

### Q5: 如何处理类别不平衡？

**A**: 有两种方法：

1. **SMOTE 过采样** (训练时):
   ```bash
   --minority_oversample yes
   ```

2. **使用平衡准确率** (评估时):
   代码自动计算 `weighted_accuracy` (balanced accuracy)

对于本项目，数据集已经相对平衡，通常不需要 SMOTE。

### Q6: 如何选择最佳模型？

**A**: 模型选择基于 **最低 dev loss**：

```python
if dev_loss < best_dev_loss:
    best_model = copy.deepcopy(model)
    best_dev_loss = dev_loss
    # ... 保存其他 dev 指标
```

你也可以修改代码，基于 `dev_auroc` 或 `dev_f1` 选择：

```python
if dev_auroc > best_dev_auroc:  # 改为最大化 AUROC
    best_model = copy.deepcopy(model)
```

### Q7: 训练多久合适？

**A**: 根据模态和数据量：

| 模态 | 推荐 epochs | 训练时间 (GPU) |
|------|-------------|----------------|
| Finger | 50-100 | ~5-10 分钟 |
| Quick | 50-80 | ~10-20 分钟 |
| Smile | 50-100 | ~10-20 分钟 |

使用学习率调度器可以帮助模型在后期收敛更好。

### Q8: 如何对比真实数据 vs 合成数据训练的模型？

**A**: 

1. **修改 `MODEL_TAG`**，避免覆盖：

```python
# 真实数据
MODEL_TAG = "real_data_baal"

# 合成数据
MODEL_TAG = "synthetic_data_baal"
```

2. **使用相同的超参数和 seed**
3. **在 wandb 中对比结果**

### Q9: GPU 内存不足怎么办？

**A**:

1. **减小 batch_size**:
   ```bash
   --batch_size 128  # 从 256/512 减少
   ```

2. **减少 num_trials**:
   ```bash
   --num_trials 300  # 从 5000 减少
   ```

3. **使用 CPU** (编辑脚本，注释掉 GPU 选择代码)

### Q10: 如何训练特定 scenario 的单模态模型？

**A**:

1. **创建新的 constants 文件**，例如 `constants_scenario_4.py`:

```python
import os

BASE_DIR = os.getcwd()+"/../../../"

FEATURES_FILE = os.path.join(BASE_DIR,"data/aligned_synthetic_format/mixed_scenario_4_real_finger_quick/features_demography_diagnosis.csv")

MODEL_TAG = "scenario_4_baal"
MODEL_BASE_PATH = os.path.join(BASE_DIR,f"models/finger_model_{MODEL_TAG}")
```

2. **修改训练脚本的 import**:

```python
# from constants_baal import *
from constants_scenario_4 import *
```

3. **运行训练**:

```bash
python unimodal_finger_baal.py --seed 0
```

---

## 快速开始示例

### 示例 1: 训练 Finger Tapping 模型（真实数据）

```bash
# 1. 进入目录
cd /localdisk2/pliu/park_multitask_fusion-main/code/unimodal_models/finger_tapping

# 2. 编辑 constants_baal.py，确保使用真实数据
# FEATURES_FILE = os.path.join(BASE_DIR,"data/finger_tapping/features_demography_diagnosis_Nov22_2023.csv")

# 3. 运行训练
python unimodal_finger_baal.py \
  --seed 42 \
  --num_epochs 50 \
  --batch_size 256 \
  --learning_rate 0.5

# 4. 查看结果
# 模型保存在: models/finger_model_both_hand_fusion_baal/predictive_model/model.pth
# Wandb 链接会在训练开始时打印
```

### 示例 2: 训练 Quick 模型（合成数据）

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/unimodal_models/quick_brown_fox

# 确保 constants_baal.py 中使用合成数据
# WAVLM_FEATURES_FILE = os.path.join(BASE_PATH,"data/synthetic_data/merged/quick_embeddings.csv")

nohup python unimodal_fox_baal.py --seed 42 --num_epochs 50 > training.log 2>&1 &

# 监控训练
tail -f training.log
```

### 示例 3: 批量训练多个 seeds

创建批量训练脚本 `train_multiple_seeds.sh`:

```bash
#!/bin/bash

cd /localdisk2/pliu/park_multitask_fusion-main/code/unimodal_models/finger_tapping

for seed in {0..10}
do
  echo "Training with seed $seed"
  python unimodal_finger_baal.py \
    --seed $seed \
    --num_epochs 50 \
    --batch_size 256
done
```

运行：
```bash
chmod +x train_multiple_seeds.sh
nohup ./train_multiple_seeds.sh > multi_seed_training.log 2>&1 &
```

---

## 总结

### 训练流程

1. ✅ **配置数据路径** (`constants_baal.py`)
2. ✅ **选择超参数** (命令行参数)
3. ✅ **运行训练** (`python unimodal_xxx_baal.py`)
4. ✅ **监控进度** (wandb 或日志文件)
5. ✅ **评估结果** (test set metrics)
6. ✅ **保存模型** (自动保存到 models/)

### 推荐工作流

```bash
# Step 1: 单次训练测试
python unimodal_finger_baal.py --seed 0 --num_epochs 10

# Step 2: 完整训练
nohup python unimodal_finger_baal.py --seed 42 --num_epochs 73 > training.log 2>&1 &

# Step 3: 超参数搜索
wandb sweep sweep_config.yaml
wandb agent <sweep_id>

# Step 4: 最佳模型重训练（多个 seeds）
for seed in {0..100}; do
  python unimodal_finger_baal.py --seed $seed --num_epochs 73
done
```

---

**文档创建时间**: 2024-12-22  
**代码库版本**: park_multitask_fusion-main  
**作者**: Generated from codebase analysis

