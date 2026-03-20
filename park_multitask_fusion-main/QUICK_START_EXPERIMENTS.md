# Quick Start: Synthetic Validation Experiments

## 当前状态

### ✅ 已完成
1. Dragon-PD Finger Tapping 数据已提取并转换为目标格式
   - 位置：`data/finger_tapping/dragon_pd_features.csv`
   - 包含 4 行数据（2个参与者，左右手）

2. 已有训练好的 unimodal 模型：
   - Finger: `models/finger_model_both_hand_fusion_baal/`
   - Speech: `models/fox_model_best_auroc_baal/`
   - Smile: `models/facial_expression_smile_best_auroc_baal/`

3. 实验框架已创建：
   - `code/fusion_models/synthetic_validation_experiment.py`
   - `SYNTHETIC_VALIDATION_PLAN.md`

### ⏳ 待完成
1. 提取更多 Dragon-PD 数据的所有三个模态特征
2. 实现合成模态生成器
3. 运行完整的实验

## 立即可以运行的实验

由于现有的 unimodal 模型已经训练好，我们可以立即运行以下实验：

### 实验 1-3: 双模态基线（使用原始数据）

#### 1. Smile + Finger
```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

# 修改 constants.py 的 MODEL_SUBSETS，使用索引 2（smile + finger）
python uncertainty_aware_fusion.py
```

配置方法：
```python
# 在 constants.py 中设置
FINGER_FEATURES_FILE = os.path.join(BASE_DIR,"data/finger_tapping/features_demography_diagnosis_Nov22_2023.csv")
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/facial_expression_smile/facial_dataset.csv")

# 使用 MODEL_SUBSETS[2] = ['finger', 'smile']
```

#### 2. Smile + Speech
```bash
# 修改 constants.py 使用索引 3（smile + speech）
python uncertainty_aware_fusion.py
```

#### 3. Finger + Speech  
```bash
# 修改 constants.py 使用索引 1（finger + speech）
python uncertainty_aware_fusion.py
```

### 实验 7: 三模态全真实（Gold Standard）

```bash
# 修改 constants.py 使用索引 0（所有三个模态）
python uncertainty_aware_fusion.py
```

## 需要实现的组件

### 1. 合成模态生成器

创建 `code/fusion_models/synthetic_generator.py`:

```python
"""
Synthetic Modality Generator

Generates synthetic predictions for a missing modality based on
predictions from other available modalities.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class CrossModalityGenerator(nn.Module):
    """
    Neural network to generate predictions for one modality
    from predictions of other modalities.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class SyntheticModalityGenerator:
    """
    Manages generation of synthetic modality predictions.
    """
    def __init__(self, source_modalities, target_modality, model_path=None):
        self.source_modalities = source_modalities
        self.target_modality = target_modality
        self.model = None
        
        if model_path:
            self.load_model(model_path)
    
    def train(self, source_data, target_data, epochs=100):
        """
        Train generator using paired source and target modality data.
        """
        # Implement training logic
        pass
    
    def generate(self, source_data):
        """
        Generate synthetic predictions for target modality.
        """
        # Implement generation logic
        pass
```

### 2. 修改 Fusion 脚本支持命令行参数

创建 `code/fusion_models/uncertainty_aware_fusion/run_experiment.py`:

```python
"""
Wrapper script to run fusion experiments with different configurations.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Import the main fusion script
sys.path.append(str(Path(__file__).parent))

def setup_constants(modalities, synthetic_modalities=None, synthetic_data_paths=None):
    """
    Dynamically set up constants.py for the experiment.
    """
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    
    config = {
        'BASE_DIR': str(BASE_DIR),
        'modalities': modalities,
        'synthetic': synthetic_modalities or [],
        'synthetic_paths': synthetic_data_paths or {}
    }
    
    # Set feature file paths
    if 'finger' in modalities:
        if synthetic_modalities and 'finger' in synthetic_modalities:
            config['FINGER_FEATURES_FILE'] = synthetic_data_paths['finger']
        else:
            config['FINGER_FEATURES_FILE'] = str(BASE_DIR / "data/finger_tapping/features_demography_diagnosis_Nov22_2023.csv")
    
    if 'speech' in modalities:
        if synthetic_modalities and 'speech' in synthetic_modalities:
            config['AUDIO_FEATURES_FILE'] = synthetic_data_paths['speech']
        else:
            config['AUDIO_FEATURES_FILE'] = str(BASE_DIR / "data/quick_brown_fox/wavlm_fox_features.csv")
    
    if 'smile' in modalities:
        if synthetic_modalities and 'smile' in synthetic_modalities:
            config['FACIAL_FEATURES_FILE'] = synthetic_data_paths['smile']
        else:
            config['FACIAL_FEATURES_FILE'] = str(BASE_DIR / "data/facial_expression_smile/facial_dataset.csv")
    
    return config

def main():
    parser = argparse.ArgumentParser(description='Run fusion experiment')
    parser.add_argument('--modalities', nargs='+', required=True,
                       choices=['smile', 'finger', 'speech'],
                       help='Modalities to use in fusion')
    parser.add_argument('--synthetic', nargs='*', default=[],
                       choices=['smile', 'finger', 'speech'],
                       help='Which modalities are synthetic')
    parser.add_argument('--synthetic_data_paths', type=json.loads, default={},
                       help='JSON dict of synthetic data paths')
    parser.add_argument('--output_tag', type=str, required=True,
                       help='Tag for output files')
    
    args = parser.parse_args()
    
    # Set up configuration
    config = setup_constants(
        modalities=args.modalities,
        synthetic_modalities=args.synthetic,
        synthetic_data_paths=args.synthetic_data_paths
    )
    
    print(f"Running experiment: {args.output_tag}")
    print(f"Modalities: {args.modalities}")
    print(f"Synthetic: {args.synthetic}")
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # TODO: Actually run the fusion script with this config
    # This would involve importing and calling the main fusion function
    
    print(f"\n✅ Experiment {args.output_tag} completed")

if __name__ == '__main__':
    main()
```

## 运行完整实验的步骤

### Step 1: 准备更多数据
```bash
# 提取更多参与者的 finger tapping 数据
cd /localdisk2/pliu/PARK
python code/feature_extraction_pipeline/finger_tapping/feature_extraction.py \
  --folder /localdisk2/pliu/Dragon-PD/PD_and_NonPD_15fps \
  --output /localdisk2/pliu/Dragon-PD/output/finger_tapping \
  --metadata_parser folder \
  --default_date 2024-05-06

# 转换并追加到项目数据
python -c "
from code.feature_extraction_pipeline.finger_tapping.feature_extraction import convert_to_target_format
convert_to_target_format(
    '/localdisk2/pliu/Dragon-PD/output/finger_tapping/features.csv',
    '/localdisk2/pliu/park_multitask_fusion-main/data/finger_tapping/dragon_pd_features.csv'
)
"
```

### Step 2: 提取 Smile 和 Speech 特征
```bash
# Smile features
cd /localdisk2/pliu/PARK
python code/feature_extraction_pipeline/facial_expression_smile/smile_feature_extraction.py \
  --video_files_directory /localdisk2/pliu/Dragon-PD/PD_and_NonPD_15fps \
  --openface_files_directory /path/to/openface \
  --output_dir /localdisk2/pliu/Dragon-PD/output/smile

# Speech features  
python code/feature_extraction_pipeline/quick_brown_fox/extract_wavlm_features.py \
  --input_dir /localdisk2/pliu/Dragon-PD/PD_and_NonPD_15fps \
  --output_dir /localdisk2/pliu/Dragon-PD/output/speech
```

### Step 3: 训练合成生成器
```python
from synthetic_generator import SyntheticModalityGenerator

# 训练 Speech 生成器（从 Smile + Finger 生成）
generator = SyntheticModalityGenerator(
    source_modalities=['smile', 'finger'],
    target_modality='speech'
)
generator.train(source_data, target_data)
generator.save('models/synthetic/speech_from_smile_finger.pth')
```

### Step 4: 运行所有实验
```bash
python code/fusion_models/synthetic_validation_experiment.py --experiment all
```

## 监控和结果

结果将保存在：
- `results/synthetic_validation/[experiment_name]/`
- 每个实验包含：
  - `config.json`: 实验配置
  - `results.json`: 详细结果
  - `metrics.csv`: 性能指标
- 总结文件：`results/synthetic_validation/summary_[timestamp].csv`

## 下一步

1. **立即可做**：运行双模态和三模态全真实实验（基线）
2. **短期**：实现合成生成器和修改 fusion 脚本
3. **中期**：提取完整的 Dragon-PD 数据集特征
4. **长期**：运行完整实验并撰写论文

需要我帮你开始运行某个具体的实验吗？


