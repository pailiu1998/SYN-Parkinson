# 中文数据单模态测试配置文件

## 📋 概述

为三个模态创建了专门的中文数据测试配置文件，用于在中文数据上进行**inference only**测试（使用已训练好的模型，不重新训练）。

---

## 📦 配置文件位置

| 模态 | 配置文件路径 |
|------|-------------|
| **Finger** | `finger_tapping/constants_baal_chinese.py` |
| **Smile** | `facial_expression_smile/constants_baal_chinese.py` |
| **Speech** | `quick_brown_fox/constants_baal_chinese.py` |

---

## 📊 数据文件映射

所有数据文件位于: `/localdisk2/pliu/park_multitask_fusion-main/data/chinese_synthetic_data/real_chinese_smile_finger/processed/`

| 模态 | 数据文件 | 大小 | 说明 |
|------|---------|------|------|
| **Finger** | `features_demography_diagnosis_Nov22_2023.csv` | 6.6M | 合并的finger特征（中英文数据）|
| **Smile** | `facial_dataset.csv` | 1.5M | 合并的smile特征（中英文数据）|
| **Speech** | `wavlm_fox_features.csv` | 35M | 合并的speech特征（中英文数据）|

---

## 🎯 统一配置

### 所有配置文件都包含：

1. **BASE_DIR** (统一设置)
   ```python
   BASE_DIR = "/localdisk2/pliu/park_multitask_fusion-main"
   ```

2. **数据路径** (指向processed目录)
   ```python
   FEATURES_FILE = os.path.join(BASE_DIR, "data/chinese_synthetic_data/real_chinese_smile_finger/processed/...")
   ```

3. **模型路径** (使用训练好的checkpoint)
   ```python
   MODEL_BASE_PATH = os.path.join(BASE_DIR, f"models/{model_name}_{MODEL_TAG}")
   MODEL_PATH = os.path.join(MODEL_BASE_PATH, "predictive_model/model.pth")
   SCALER_PATH = os.path.join(MODEL_BASE_PATH, "scaler/scaler.pth")
   ```

---

## 🔧 具体配置详情

### 1️⃣ Finger Tapping (指头点击)

**配置文件**: `finger_tapping/constants_baal_chinese.py`

```python
BASE_DIR = "/localdisk2/pliu/park_multitask_fusion-main"
FEATURES_FILE = ".../processed/features_demography_diagnosis_Nov22_2023.csv"
MODEL_TAG = "both_hand_fusion_baal"
MODEL_BASE_PATH = ".../models/finger_model_both_hand_fusion_baal"
```

**使用的模型**: `finger_model_both_hand_fusion_baal`
- 训练数据: 英文数据
- 测试数据: 中英文合并数据（62个中文参与者）

---

### 2️⃣ Facial Expression Smile (微笑表情)

**配置文件**: `facial_expression_smile/constants_baal_chinese.py`

```python
BASE_DIR = "/localdisk2/pliu/park_multitask_fusion-main"
FEATURES_FILE = ".../processed/facial_dataset.csv"
MODEL_TAG = "best_auroc_baal"
MODEL_BASE_PATH = ".../models/facial_expression_smile_best_auroc_baal"

FACIAL_EXPRESSIONS = {
    'smile': True,
    'surprise': False,
    'disgust': False
}
```

**使用的模型**: `facial_expression_smile_best_auroc_baal`
- 训练数据: 英文数据
- 测试数据: 中英文合并数据（56个中文参与者）

---

### 3️⃣ Quick Brown Fox (语音)

**配置文件**: `quick_brown_fox/constants_baal_chinese.py`

```python
BASE_DIR = "/localdisk2/pliu/park_multitask_fusion-main"
WAVLM_FEATURES_FILE = ".../processed/wavlm_fox_features.csv"
MODEL_TAG = "best_auroc_baal"
MODEL_BASE_PATH = ".../models/fox_model_best_auroc_baal"

# 兼容旧变量名
FINGER_FEATURES_FILE = ".../features_demography_diagnosis_Nov22_2023.csv"
FACIAL_FEATURES_FILE = ".../facial_dataset.csv"
```

**使用的模型**: `fox_model_best_auroc_baal`
- 训练数据: 英文数据
- 测试数据: 中英文合并数据

**特别说明**: 此配置文件还包含了所有三个模态的数据路径，方便多模态使用。

---

## 🚀 如何使用

### 运行单模态测试

根据需要修改对应的训练脚本，导入中文配置：

#### Finger测试
```python
# 在 train_chinese_finger.py 中
from constants_baal_chinese import *
# 然后运行inference部分
```

#### Smile测试
```python
# 在 train_chinese_smile.py 中
from constants_baal_chinese import *
# 然后运行inference部分
```

#### Speech测试
```python
# 在对应的测试脚本中
from constants_baal_chinese import *
# 然后运行inference部分
```

---

## 📊 数据集划分

使用的participant划分文件：
```
data/chinese_synthetic_data/real_chinese_smile_finger/processed/
├── dev_set_participants.txt    (330 participants)
└── test_set_participants.txt   (63 participants)
```

---

## ✅ 验证配置

### 检查配置文件
```bash
# 查看所有配置文件
ls -lh finger_tapping/constants_baal_chinese.py
ls -lh facial_expression_smile/constants_baal_chinese.py
ls -lh quick_brown_fox/constants_baal_chinese.py
```

### 检查数据文件
```bash
# 验证数据文件存在
ls -lh /localdisk2/pliu/park_multitask_fusion-main/data/chinese_synthetic_data/real_chinese_smile_finger/processed/*.csv
```

### Python验证
```python
# Finger
from finger_tapping.constants_baal_chinese import *
print(f"Finger data: {FEATURES_FILE}")
print(f"Model: {MODEL_BASE_PATH}")

# Smile
from facial_expression_smile.constants_baal_chinese import *
print(f"Smile data: {FEATURES_FILE}")
print(f"Model: {MODEL_BASE_PATH}")

# Speech
from quick_brown_fox.constants_baal_chinese import *
print(f"Speech data: {WAVLM_FEATURES_FILE}")
print(f"Model: {MODEL_BASE_PATH}")
```

---

## 🎯 与原始配置的区别

| 项目 | 原始配置 | 中文配置 |
|------|---------|---------|
| **BASE_DIR** | `os.getcwd()+"/../../../"` | `/localdisk2/pliu/park_multitask_fusion-main` |
| **数据路径** | 相对路径 | 绝对路径，指向processed目录 |
| **数据来源** | 英文数据 / 合成数据 | 中英文合并真实数据 |
| **用途** | 训练 + 测试 | 仅测试（inference） |

---

## 📈 预期使用场景

1. **单模态测试**: 在中文数据上测试每个模态的性能
2. **跨语言泛化**: 评估英文训练模型在中文数据上的表现
3. **数据质量检查**: 验证中文数据的特征提取是否正确
4. **baseline性能**: 为多模态融合提供单模态baseline

---

## 🔗 相关文档

- 双模态融合配置: `fusion_models/uncertainty_aware_fusion/constants_chinese_bimodal.py`
- 双模态融合脚本: `fusion_models/uncertainty_aware_fusion/run_chinese_bimodal_fusion.py`
- 双模态融合结果: `fusion_models/uncertainty_aware_fusion/CHINESE_BIMODAL_RESULTS.md`

---

## 📝 注意事项

1. ⚠️ **BASE_DIR必须是绝对路径**: 确保在任何目录下运行都能找到正确的文件
2. ⚠️ **模型checkpoint**: 确保对应的模型文件存在于 `models/` 目录
3. ⚠️ **数据格式**: 确保中文数据的列名和格式与训练数据一致
4. ⚠️ **participant划分**: 使用processed目录下的划分文件，不是原始的txt文件

---

**创建日期**: 2026-01-20  
**状态**: ✅ 所有配置文件已创建并验证

