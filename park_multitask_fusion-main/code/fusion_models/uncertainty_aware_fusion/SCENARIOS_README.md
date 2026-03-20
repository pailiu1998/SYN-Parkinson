# Fusion Model Scenarios

本项目包含4个不同的实验场景，用于测试模型对合成数据的泛化能力。

## Baseline（基线）
- **配置文件**: `constants.py`
- **运行脚本**: `uncertainty_aware_fusion.py`
- **数据**:
  - Finger: 真实数据
  - Quick: 真实数据
  - Smile: 真实数据
- **目的**: 作为对比基线，全部使用真实数据

## Scenario 1: 真实 Finger + 合成 Quick + 合成 Smile
- **配置文件**: `constants_scenario_1.py`
- **运行脚本**: `uncertainty_aware_fusion_scenario_1.py`
- **数据路径**: `data/aligned_synthetic_format/mixed_scenario_1_real_finger/`
- **数据**:
  - Finger: 真实数据（train/dev/test全部真实）
  - Quick: train/dev用真实，test用合成
  - Smile: train/dev用真实，test用合成
- **目的**: 测试模型对合成 Quick 和 Smile 数据的泛化能力

## Scenario 2: 合成 Finger + 合成 Quick + 真实 Smile
- **配置文件**: `constants_scenario_2.py`
- **运行脚本**: `uncertainty_aware_fusion_scenario_2.py`
- **数据路径**: `data/aligned_synthetic_format/mixed_scenario_2_real_smile/`
- **数据**:
  - Finger: train/dev用真实，test用合成
  - Quick: train/dev用真实，test用合成
  - Smile: 真实数据（train/dev/test全部真实）
- **目的**: 测试模型对合成 Finger 和 Quick 数据的泛化能力

## Scenario 3: 合成 Finger + 真实 Quick + 合成 Smile
- **配置文件**: `constants_scenario_3.py`
- **运行脚本**: `uncertainty_aware_fusion_scenario_3.py`
- **数据路径**: `data/aligned_synthetic_format/mixed_scenario_3_real_quick/`
- **数据**:
  - Finger: train/dev用真实，test用合成
  - Quick: 真实数据（train/dev/test全部真实）
  - Smile: train/dev用真实，test用合成
- **目的**: 测试模型对合成 Finger 和 Smile 数据的泛化能力

## 运行方法

```bash
# Baseline
python uncertainty_aware_fusion.py

# Scenario 1
python uncertainty_aware_fusion_scenario_1.py

# Scenario 2
python uncertainty_aware_fusion_scenario_2.py

# Scenario 3
python uncertainty_aware_fusion_scenario_3.py
```

## 数据集划分
- **Train set**: 不在 dev_set_participants.txt 和 test_set_participants.txt 中的患者
- **Dev set**: dev_set_participants.txt 中的267个患者
- **Test set**: test_set_participants.txt 中的267个患者

## 合成数据说明
- 合成数据仅用于 **test set**
- Train 和 Dev 始终使用真实数据
- 这样可以评估模型对合成数据的泛化能力





