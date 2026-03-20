# VAE vs Diffusion Synthetic Data Validation Experiments

## 实验设计

### 6个实验

**VAE合成数据（3个）：**
1. `vae_exp4`: Real Smile + Real Finger + **VAE Synthetic Speech**
2. `vae_exp5`: Real Smile + **VAE Synthetic Finger** + Real Speech
3. `vae_exp6`: **VAE Synthetic Smile** + Real Finger + Real Speech

**Diffusion合成数据（3个）：**
4. `diff_exp4`: Real Smile + Real Finger + **Diffusion Synthetic Speech**
5. `diff_exp5`: Real Smile + **Diffusion Synthetic Finger** + Real Speech
6. `diff_exp6`: **Diffusion Synthetic Smile** + Real Finger + Real Speech

## 数据准备

### VAE数据（已准备好）
位置：`data/synthetic_data/vae_synthetic/`
- `facial_dataset.csv` (Smile)
- `wavlm_fox_features.csv` (Speech)
- `features_demography_diagnosis.csv` (Finger)

### Diffusion数据（待提供）
位置：`data/synthetic_data/diffusion_synthetic/`
需要包含3个文件：
- `facial_dataset.csv` (Smile)
- `wavlm_fox_features.csv` (Speech)
- `features_demography_diagnosis.csv` (Finger)

## 使用方法

### 1. 准备Diffusion数据
将diffusion合成数据放到：
```
data/synthetic_data/diffusion_synthetic/
├── facial_dataset.csv
├── wavlm_fox_features.csv
└── features_demography_diagnosis.csv
```

### 2. 运行所有实验
```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

# 运行所有6个实验
python run_vae_diffusion_experiments.py --experiment all

# 只运行VAE实验
python run_vae_diffusion_experiments.py --method vae

# 只运行Diffusion实验
python run_vae_diffusion_experiments.py --method diffusion

# 运行单个实验
python run_vae_diffusion_experiments.py --experiment vae_exp4_smile_finger_synth_speech
```

### 3. 查看结果
结果保存在：
```
results/synthetic_validation/vae_diffusion_comparison/
├── vae_exp4_smile_finger_synth_speech.log
├── vae_exp5_smile_synth_finger_speech.log
├── vae_exp6_synth_smile_finger_speech.log
├── diff_exp4_smile_finger_synth_speech.log
├── diff_exp5_smile_synth_finger_speech.log
└── diff_exp6_synth_smile_finger_speech.log
```

## 实验流程

脚本会自动：
1. ✅ 检查数据文件是否存在
2. ✅ 为每个实验生成对应的`constants.py`配置
3. ✅ 运行fusion模型训练和评估
4. ✅ 保存日志和结果
5. ✅ 提取AUROC等关键指标

## 预期输出

每个实验会输出：
- AUROC
- Accuracy
- F1 Score
- Precision/Recall
- 其他评估指标

## 注意事项

- 确保conda环境`park`已激活或可用
- 每个实验大约需要5-10分钟
- 所有6个实验总共需要30-60分钟
- 结果会自动保存到wandb和本地日志文件

## 快速开始

```bash
# 1. 确保diffusion数据已准备好
ls data/synthetic_data/diffusion_synthetic/*.csv

# 2. 运行所有实验
python run_vae_diffusion_experiments.py --experiment all

# 3. 查看结果摘要
grep -r "auroc" results/synthetic_validation/vae_diffusion_comparison/*.log
```


