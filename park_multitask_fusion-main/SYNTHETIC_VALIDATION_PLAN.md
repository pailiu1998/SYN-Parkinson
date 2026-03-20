# Synthetic Data Validation Experiment Plan

## 目标
验证合成模态数据的有效性，通过比较不同模态组合的融合模型性能。

## 实验设计

### 数据集
- **原始数据集**：park_multitask_fusion-main 中的真实数据
  - Smile (facial_expression_smile)
  - Finger Tapping (finger_tapping)
  - Speech (quick_brown_fox)

- **合成数据集**：Dragon-PD 数据集（待生成所有模态）
  - 当前已有：Finger Tapping 特征
  - 待生成：Smile 和 Speech 特征

### 实验组（7组）

#### 组1-3：双模态基线（Two-Modality Baseline）
用于建立基线性能，了解任意两个模态的融合效果。

1. **Smile + Finger**
   - 模型：两个真实模态的 unimodal 模型
   - Fusion：Late fusion with uncertainty
   
2. **Smile + Speech**
   - 模型：两个真实模态的 unimodal 模型
   - Fusion：Late fusion with uncertainty
   
3. **Finger + Speech**
   - 模型：两个真实模态的 unimodal 模型
   - Fusion：Late fusion with uncertainty

#### 组4-6：两真实+一合成（Two-Real + One-Synthetic）
验证合成数据能否有效补充真实数据。

4. **Smile + Finger + Synthetic Speech**
   - 真实：Smile, Finger
   - 合成：Speech（从 Smile 和/或 Finger 生成）
   
5. **Smile + Synthetic Finger + Speech**
   - 真实：Smile, Speech
   - 合成：Finger（从 Smile 和/或 Speech 生成）
   
6. **Synthetic Smile + Finger + Speech**
   - 真实：Finger, Speech
   - 合成：Smile（从 Finger 和/或 Speech 生成）

#### 组7：金标准（Gold Standard）
所有真实数据，作为性能上限参考。

7. **Smile + Finger + Speech (All Real)**
   - 所有三个模态都使用真实数据

## 实验步骤

### Phase 1: 准备工作
- [x] 1.1 生成 Finger Tapping 特征（Dragon-PD 数据）
- [ ] 1.2 生成 Smile 特征（Dragon-PD 数据）
- [ ] 1.3 生成 Speech 特征（Dragon-PD 数据）
- [ ] 1.4 转换所有特征为目标格式
- [ ] 1.5 准备合成数据生成模型

### Phase 2: 训练 Unimodal 模型
使用原始数据集训练三个 unimodal 模型：

```bash
# Smile unimodal
cd code/unimodal_models/facial_expression_smile
python unimodal_smile_baal.py

# Finger unimodal  
cd code/unimodal_models/finger_tapping
python unimodal_finger_baal.py

# Speech unimodal
cd code/unimodal_models/quick_brown_fox
python unimodal_fox_wavlm_baal.py
```

### Phase 3: 运行双模态实验（基线）

#### 实验 1: Smile + Finger
```bash
cd code/fusion_models/uncertainty_aware_fusion
# 修改 constants.py 使用 smile 和 finger 模态
python uncertainty_aware_fusion.py \
  --modalities smile finger \
  --output_tag 2mod_smile_finger
```

#### 实验 2: Smile + Speech
```bash
python uncertainty_aware_fusion.py \
  --modalities smile speech \
  --output_tag 2mod_smile_speech
```

#### 实验 3: Finger + Speech
```bash
python uncertainty_aware_fusion.py \
  --modalities finger speech \
  --output_tag 2mod_finger_speech
```

### Phase 4: 生成合成模态预测

对于每个缺失的模态，使用训练好的生成模型创建合成预测：

```python
# 示例：从 Smile 和 Finger 生成 Speech 预测
from synthetic_generator import SyntheticModalityGenerator

generator = SyntheticModalityGenerator(
    source_modalities=['smile', 'finger'],
    target_modality='speech'
)

synthetic_speech_preds = generator.generate(
    smile_features=smile_features,
    finger_features=finger_features
)
```

### Phase 5: 运行三模态实验

#### 实验 4: Smile + Finger + Synthetic Speech
```bash
python uncertainty_aware_fusion.py \
  --modalities smile finger speech \
  --synthetic speech \
  --synthetic_data_path data/synthetic/speech_from_smile_finger.csv \
  --output_tag 3mod_smile_finger_synth_speech
```

#### 实验 5: Smile + Synthetic Finger + Speech
```bash
python uncertainty_aware_fusion.py \
  --modalities smile finger speech \
  --synthetic finger \
  --synthetic_data_path data/synthetic/finger_from_smile_speech.csv \
  --output_tag 3mod_smile_synth_finger_speech
```

#### 实验 6: Synthetic Smile + Finger + Speech
```bash
python uncertainty_aware_fusion.py \
  --modalities smile finger speech \
  --synthetic smile \
  --synthetic_data_path data/synthetic/smile_from_finger_speech.csv \
  --output_tag 3mod_synth_smile_finger_speech
```

#### 实验 7: All Real (Gold Standard)
```bash
python uncertainty_aware_fusion.py \
  --modalities smile finger speech \
  --output_tag 3mod_all_real
```

### Phase 6: 评估和比较

收集所有实验的指标：
- AUROC
- Accuracy
- F1 Score
- Precision / Recall
- Brier Score
- Confusion Matrix

## 评估指标

### 主要指标
1. **AUROC**: 主要性能指标
2. **Accuracy**: 分类准确率
3. **F1 Score**: 平衡精确率和召回率
4. **Precision/Recall**: 查准率和查全率

### 比较分析
1. **双模态 vs 三模态（含合成）**: 验证合成数据是否有增益
2. **三模态（含合成）vs 三模态（全真实）**: 评估合成数据与真实数据的差距
3. **不同合成模态的效果**: 哪个模态最容易/难合成

### 统计显著性检验
- 使用配对 t 检验比较不同组合
- 计算置信区间
- Bootstrap 重采样验证稳定性

## 预期结果

### 假设 H1: 合成数据有效性
三模态（含一个合成）的性能 > 双模态（全真实）的性能

如果成立，说明合成数据能够有效补充缺失模态。

### 假设 H2: 合成数据接近真实
三模态（含一个合成）的性能 ≈ 三模态（全真实）的性能（差距 < 5%）

如果成立，说明合成数据质量接近真实数据。

## 文件组织

```
park_multitask_fusion-main/
├── results/
│   └── synthetic_validation/
│       ├── 2mod_smile_finger/
│       │   ├── config.json
│       │   ├── results.json
│       │   └── metrics.csv
│       ├── 2mod_smile_speech/
│       ├── 2mod_finger_speech/
│       ├── 3mod_smile_finger_synth_speech/
│       ├── 3mod_smile_synth_finger_speech/
│       ├── 3mod_synth_smile_finger_speech/
│       ├── 3mod_all_real/
│       └── summary_YYYYMMDD_HHMMSS.csv
├── data/
│   └── synthetic/
│       ├── speech_from_smile_finger.csv
│       ├── finger_from_smile_speech.csv
│       └── smile_from_finger_speech.csv
└── code/
    └── fusion_models/
        └── synthetic_validation_experiment.py
```

## 使用说明

### 列出所有实验
```bash
python code/fusion_models/synthetic_validation_experiment.py --list
```

### 运行单个实验
```bash
python code/fusion_models/synthetic_validation_experiment.py --experiment 2mod_smile_finger
```

### 运行所有实验
```bash
python code/fusion_models/synthetic_validation_experiment.py --experiment all
```

## 下一步工作

1. **立即可做**：
   - [ ] 完成 Smile 和 Speech 特征提取（Dragon-PD 数据）
   - [ ] 检查现有 unimodal 模型是否已训练好
   - [ ] 实现 fusion 脚本的命令行参数

2. **需要实现**：
   - [ ] 合成模态生成器（从两个模态生成第三个）
   - [ ] 修改 fusion 脚本支持合成数据输入
   - [ ] 自动化实验运行和结果收集

3. **分析和可视化**：
   - [ ] 创建性能对比图表
   - [ ] 统计显著性分析
   - [ ] 撰写实验报告


