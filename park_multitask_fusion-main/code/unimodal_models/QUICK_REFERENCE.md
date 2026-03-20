# 单模态训练快速参考

## 🚀 快速开始

### 一键训练所有模态

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/unimodal_models
chmod +x train_all_modalities.sh
./train_all_modalities.sh
```

### 单独训练各模态

```bash
# Finger Tapping
cd finger_tapping
python unimodal_finger_baal.py --seed 42 --num_epochs 73

# Quick Brown Fox
cd quick_brown_fox
python unimodal_fox_baal.py --seed 42 --num_epochs 55

# Facial Expression
cd facial_expression_smile
python unimodal_smile_baal.py --seed 42 --num_epochs 50
```

---

## 📊 模态对比表

| 特性 | Finger Tapping | Quick Brown Fox | Facial Expression |
|------|----------------|-----------------|-------------------|
| **特征类型** | 运动特征 | WavLM 音频嵌入 | 面部特征 |
| **特征维度** | ~120-250 | 1024 | ~50 |
| **训练样本数** | ~2,000-3,000 | ~1,300-1,800 | ~1,500-1,700 |
| **推荐 epochs** | 73 | 55 | 50 |
| **推荐 batch_size** | 512 | 256 | 256 |
| **num_trials** | 300 | 5000 | 5000 |
| **训练时间** | ~5-10 分钟 | ~10-20 分钟 | ~10-20 分钟 |
| **特征标准化** | ✅ 需要 | ❌ 不需要 | ✅ 需要 |
| **SMOTE 过采样** | ❌ 不需要 | ✅ 需要 | ❌ 不需要 |
| **删除相关特征** | ❌ 不需要 | ❌ 不需要 | ✅ 需要 |

---

## 🎯 最佳超参数（已优化）

### Finger Tapping

```bash
python unimodal_finger_baal.py \
  --model ShallowANN \
  --dropout_prob 0.14 \
  --learning_rate 0.67 \
  --batch_size 512 \
  --num_epochs 73 \
  --momentum 0.84 \
  --use_scheduler yes \
  --scheduler step \
  --step_size 22 \
  --gamma 0.66 \
  --hand both \
  --use_feature_scaling yes
```

### Quick Brown Fox

```bash
python unimodal_fox_baal.py \
  --model ShallowANN \
  --dropout_prob 0.08 \
  --learning_rate 0.93 \
  --batch_size 256 \
  --num_epochs 55 \
  --momentum 0.49 \
  --use_feature_scaling no \
  --minority_oversample yes \
  --use_scheduler no
```

### Facial Expression

```bash
python unimodal_smile_baal.py \
  --model ShallowANN \
  --dropout_prob 0.1 \
  --learning_rate 0.5 \
  --batch_size 256 \
  --num_epochs 50 \
  --momentum 0.9 \
  --use_feature_scaling yes \
  --drop_correlated yes \
  --corr_thr 0.85 \
  --use_scheduler yes \
  --scheduler reduce \
  --patience 10
```

---

## 📁 数据路径配置

### 真实数据

```python
# Finger Tapping (constants_baal.py)
FEATURES_FILE = os.path.join(BASE_DIR,"data/finger_tapping/features_demography_diagnosis_Nov22_2023.csv")

# Quick Brown Fox (constants_baal.py)
WAVLM_FEATURES_FILE = os.path.join(BASE_PATH,"data/quick_brown_fox/wavlm_fox_features.csv")

# Facial Expression (constants_baal.py)
FEATURES_FILE = os.path.join(BASE_PATH,"data/facial_expression_smile/facial_dataset.csv")
```

### 合成数据

```python
# Finger Tapping
FEATURES_FILE = os.path.join(BASE_DIR,"data/synthetic_data/merged/finger_embeddings.csv")

# Quick Brown Fox
WAVLM_FEATURES_FILE = os.path.join(BASE_PATH,"data/synthetic_data/merged/quick_embeddings.csv")

# Facial Expression
FEATURES_FILE = os.path.join(BASE_PATH,"data/synthetic_data/merged/smile_embeddings.csv")
```

### Scenario 数据

```python
# Scenario 4: real_finger_quick (真实 Finger + Quick, 混合 Smile)
FEATURES_FILE = os.path.join(BASE_DIR,"data/aligned_synthetic_format/mixed_scenario_4_real_finger_quick/features_demography_diagnosis.csv")
WAVLM_FEATURES_FILE = os.path.join(BASE_PATH,"data/aligned_synthetic_format/mixed_scenario_4_real_finger_quick/wavlm_fox_features.csv")
FEATURES_FILE = os.path.join(BASE_PATH,"data/aligned_synthetic_format/mixed_scenario_4_real_finger_quick/facial_dataset.csv")

# Scenario 5: real_smile_finger (真实 Smile + Finger, 混合 Quick)
# Scenario 6: real_quick_smile (真实 Quick + Smile, 混合 Finger)
# ... 类似修改
```

---

## 🔍 常用命令

### 后台训练

```bash
# 单个模态
nohup python unimodal_finger_baal.py --seed 42 > training.log 2>&1 &

# 查看进程
ps aux | grep unimodal | grep python

# 监控日志
tail -f training.log

# 停止训练
kill <PID>
```

### 批量训练（多 seeds）

```bash
# 使用 Python 脚本
python train_multiple_seeds.py --modality finger --start_seed 0 --end_seed 10

# 使用 bash 循环
for seed in {0..10}; do
  python unimodal_finger_baal.py --seed $seed
done
```

### wandb 管理

```bash
# 登录
wandb login

# 创建 sweep
wandb sweep sweep_config.yaml

# 启动 agent
wandb agent <sweep_id>

# 后台运行 agent
nohup wandb agent <sweep_id> > sweep.log 2>&1 &
```

---

## 📈 性能基准（参考）

基于真实数据的单模态模型性能：

| 模态 | AUROC | F1 Score | Balanced Accuracy |
|------|-------|----------|-------------------|
| Finger Tapping | ~0.85-0.90 | ~0.80-0.85 | ~0.80-0.85 |
| Quick Brown Fox | ~0.85-0.90 | ~0.80-0.85 | ~0.80-0.85 |
| Facial Expression | ~0.80-0.85 | ~0.75-0.80 | ~0.75-0.80 |

**注意**: 实际性能取决于数据集、超参数和随机种子。

---

## 🐛 故障排查

### GPU 内存不足

```bash
# 减小 batch_size
--batch_size 128

# 减少 num_trials
--num_trials 300
```

### 训练不收敛

```bash
# 降低学习率
--learning_rate 0.1

# 增加 epochs
--num_epochs 100

# 使用学习率调度器
--use_scheduler yes --scheduler reduce
```

### wandb 同步问题

```bash
# 离线模式
wandb offline

# 手动同步
wandb sync <run_directory>
```

### 找不到 GPU

```bash
# 检查 GPU 状态
nvidia-smi

# 设置特定 GPU
export CUDA_VISIBLE_DEVICES=0
```

---

## 📂 输出文件位置

### 模型保存

```
models/
  ├── finger_model_both_hand_fusion_baal/
  │   ├── predictive_model/
  │   │   ├── model.pth           # PyTorch 模型权重
  │   │   └── model_config.json   # 训练配置
  │   └── scaler/
  │       └── scaler.pth           # StandardScaler
  ├── fox_model_best_auroc_baal/
  └── facial_expression_smile_best_auroc_baal/
```

### wandb 日志

```
unimodal_models/
  ├── finger_tapping/wandb/
  ├── quick_brown_fox/wandb/
  └── facial_expression_smile/wandb/
```

---

## 🔗 相关文件

- **详细指南**: `UNIMODAL_TRAINING_GUIDE.md`
- **批量训练脚本**: `train_all_modalities.sh`
- **多 seed 训练**: `train_multiple_seeds.py`
- **常量配置**: `*/constants_baal.py`
- **训练脚本**: `*/unimodal_*_baal.py`

---

## 💡 最佳实践

1. ✅ **先用小 epochs 测试** (如 `--num_epochs 5`)
2. ✅ **使用固定 seed** 确保可重复性
3. ✅ **监控 wandb** 查看训练曲线
4. ✅ **保存日志文件** 用于调试
5. ✅ **定期备份模型** 到其他位置
6. ✅ **使用 nohup** 进行长时间训练
7. ✅ **批量训练前** 先测试单个 seed
8. ✅ **记录最佳超参数** 方便后续使用

---

**更新时间**: 2024-12-22  
**版本**: v1.0

