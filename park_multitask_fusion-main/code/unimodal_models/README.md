# 单模态模型训练

## 📚 文档导航

本目录包含三个单模态模型的训练代码和文档：

### 主要文档

1. **[QUICK_REFERENCE.md](./QUICK_REFERENCE.md)** ⚡
   - 快速上手指南
   - 一键命令
   - 常用参数对比表
   - **推荐首先阅读**

2. **[UNIMODAL_TRAINING_GUIDE.md](./UNIMODAL_TRAINING_GUIDE.md)** 📖
   - 完整详细的训练指南
   - 模型架构详解
   - 超参数说明
   - 问题排查

### 训练脚本

3. **[train_all_modalities.sh](./train_all_modalities.sh)** 🚀
   - 一键训练所有三个模态
   - 使用优化后的超参数
   
4. **[train_multiple_seeds.py](./train_multiple_seeds.py)** 🔄
   - 批量训练多个 seeds
   - 适合收集统计结果

---

## 🎯 三个单模态

### 1. Finger Tapping (手指敲击)
- **位置**: `finger_tapping/`
- **脚本**: `unimodal_finger_baal.py`
- **数据**: 运动特征 (~120-250 维)
- **用途**: 检测帕金森病运动障碍

### 2. Quick Brown Fox (语音)
- **位置**: `quick_brown_fox/`
- **脚本**: `unimodal_fox_baal.py`
- **数据**: WavLM 音频嵌入 (1024 维)
- **用途**: 检测帕金森病语音特征

### 3. Facial Expression (面部表情)
- **位置**: `facial_expression_smile/`
- **脚本**: `unimodal_smile_baal.py`
- **数据**: 面部特征 (~50 维)
- **用途**: 检测帕金森病面部表情变化

---

## 🚀 快速开始

### 方法 1: 一键训练所有模态（推荐）

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/unimodal_models
./train_all_modalities.sh
```

### 方法 2: 单独训练

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

### 方法 3: 批量训练（多 seeds）

```bash
python train_multiple_seeds.py --modality finger --start_seed 0 --end_seed 10
```

---

## 📊 模型特点

所有单模态模型使用：
- ✅ **ShallowANN** 或 **ANN** 架构
- ✅ **Bayesian Active Learning (BAAL)** 框架
- ✅ **MC Dropout** 用于不确定性估计
- ✅ **wandb** 实验跟踪
- ✅ 自动 **GPU 选择**
- ✅ 支持 **特征标准化**
- ✅ 支持 **SMOTE 过采样**

---

## 🎓 学习路径

1. **初学者** (5 分钟)
   - 阅读 [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)
   - 运行一个简单的训练命令测试

2. **深入学习** (30 分钟)
   - 阅读 [UNIMODAL_TRAINING_GUIDE.md](./UNIMODAL_TRAINING_GUIDE.md)
   - 了解模型架构和超参数

3. **实践** (1-2 小时)
   - 使用 `train_all_modalities.sh` 训练所有模态
   - 在 wandb 中查看结果
   - 尝试调整超参数

4. **高级** (持续)
   - 使用 wandb sweep 进行超参数搜索
   - 批量训练多个 seeds 收集统计结果
   - 对比真实数据 vs 合成数据性能

---

## 📁 目录结构

```
unimodal_models/
├── README.md                          # 本文件
├── QUICK_REFERENCE.md                 # 快速参考
├── UNIMODAL_TRAINING_GUIDE.md         # 详细指南
├── train_all_modalities.sh            # 批量训练脚本
├── train_multiple_seeds.py            # 多 seed 训练脚本
│
├── finger_tapping/
│   ├── unimodal_finger_baal.py        # 训练脚本
│   ├── constants_baal.py              # 配置文件
│   └── wandb/                         # wandb 日志
│
├── quick_brown_fox/
│   ├── unimodal_fox_baal.py           # 训练脚本
│   ├── constants_baal.py              # 配置文件
│   └── wandb/                         # wandb 日志
│
└── facial_expression_smile/
    ├── unimodal_smile_baal.py         # 训练脚本
    ├── constants_baal.py              # 配置文件
    └── wandb/                         # wandb 日志
```

---

## 💡 常见任务

### 更改数据源

编辑对应模态的 `constants_baal.py`:

```python
# 使用真实数据
FEATURES_FILE = os.path.join(BASE_DIR,"data/finger_tapping/features_demography_diagnosis_Nov22_2023.csv")

# 使用合成数据
FEATURES_FILE = os.path.join(BASE_DIR,"data/synthetic_data/merged/finger_embeddings.csv")

# 使用 scenario 数据
FEATURES_FILE = os.path.join(BASE_DIR,"data/aligned_synthetic_format/mixed_scenario_4_real_finger_quick/features_demography_diagnosis.csv")
```

### 调整超参数

通过命令行参数调整：

```bash
python unimodal_finger_baal.py \
  --seed 42 \
  --num_epochs 100 \
  --learning_rate 0.5 \
  --batch_size 256 \
  --dropout_prob 0.2
```

### 后台训练

```bash
nohup python unimodal_finger_baal.py --seed 42 > training.log 2>&1 &
```

### 监控训练

```bash
# 查看日志
tail -f training.log

# 查看进程
ps aux | grep unimodal | grep python

# 查看 wandb
# 训练开始时会打印 wandb 链接
```

---

## 📈 性能预期

基于真实数据的单模态模型性能（参考）：

| 模态 | AUROC | F1 | Balanced Acc |
|------|-------|-----|--------------|
| Finger | 0.85-0.90 | 0.80-0.85 | 0.80-0.85 |
| Quick | 0.85-0.90 | 0.80-0.85 | 0.80-0.85 |
| Smile | 0.80-0.85 | 0.75-0.80 | 0.75-0.80 |

**注意**: 实际性能取决于：
- 数据集（真实/合成/混合）
- 超参数设置
- 随机种子
- 数据预处理方法

---

## 🔗 相关资源

### 融合模型
单模态模型训练后，可以用于多模态融合：
- 位置: `../fusion_models/uncertainty_aware_fusion/`
- 文档: [融合模型指南](../fusion_models/uncertainty_aware_fusion/README.md)

### Wandb 项目
- **Finger**: `park_final_experiments`
- **Quick**: `unimodal_quick_synthetic`
- **Smile**: `park_final_experiments`

---

## 🆘 需要帮助？

1. **快速问题**: 查看 [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) 的"故障排查"部分
2. **详细问题**: 查看 [UNIMODAL_TRAINING_GUIDE.md](./UNIMODAL_TRAINING_GUIDE.md) 的"常见问题"部分
3. **代码问题**: 查看各模态目录下的 `wandb_*.log` 文件

---

## 🎉 开始训练！

选择一种方式开始：

```bash
# 快速测试（5 epochs）
cd finger_tapping
python unimodal_finger_baal.py --seed 0 --num_epochs 5

# 完整训练（所有模态）
cd /localdisk2/pliu/park_multitask_fusion-main/code/unimodal_models
./train_all_modalities.sh

# 批量实验（多 seeds）
python train_multiple_seeds.py --modality finger --start_seed 0 --end_seed 10
```

祝训练顺利！🚀

---

**最后更新**: 2024-12-22  
**维护者**: park_multitask_fusion team

