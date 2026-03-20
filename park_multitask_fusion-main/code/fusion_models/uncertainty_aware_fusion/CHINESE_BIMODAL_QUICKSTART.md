# 🇨🇳 中文数据双模态融合实验 - 快速启动指南

## 📋 实验概述

**实验名称**: 中文数据双模态融合 (Smile + Finger)

**数据说明**:
- **Smile模态**: 真实中文数据，56个参与者
- **Finger模态**: 真实中文数据，62个参与者  
- **对应ID**: 55个参与者同时拥有两个模态的数据

**模型**:
- Finger: `finger_model_both_hand_fusion_baal`
- Smile: `facial_expression_smile_best_auroc_baal`

---

## 🚀 一键启动命令

### 方式1: 使用Shell脚本（推荐）

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

# 标准运行 (244 epochs)
./RUN_CHINESE_BIMODAL.sh

# 快速测试 (5 epochs)
./RUN_CHINESE_BIMODAL.sh --test

# 后台运行
./RUN_CHINESE_BIMODAL.sh --background

# 查看所有选项
./RUN_CHINESE_BIMODAL.sh --help
```

### 方式2: 直接使用Python脚本

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

# 标准运行
python run_chinese_bimodal_fusion.py

# 自定义参数
python run_chinese_bimodal_fusion.py --seed 42 --num_epochs 244 --batch_size 64

# 快速测试
python run_chinese_bimodal_fusion.py --num_epochs 5
```

---

## 🎯 快速命令总结

### 最简单的启动方式

```bash
# 一行命令搞定（包括进入目录、运行实验）
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion && ./RUN_CHINESE_BIMODAL.sh --test
```

### 后台运行 + 实时监控

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

# 后台运行
./RUN_CHINESE_BIMODAL.sh --background

# 查看实时日志（日志文件名会显示在启动时）
tail -f chinese_bimodal_YYYYMMDD_HHMMSS.log

# 或者使用最新的日志
tail -f $(ls -t chinese_bimodal_*.log | head -1)
```

### 使用 tmux（推荐长时间运行）

```bash
# 创建 tmux session
tmux new -s chinese_bimodal

# 运行实验
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion
./RUN_CHINESE_BIMODAL.sh

# 按 Ctrl+B 然后按 D detach
# 重新连接: tmux attach -t chinese_bimodal
```

---

## ⚙️ 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seed` | 42 | 随机种子 |
| `--num_epochs` / `--epochs` | 244 | 训练轮数 |
| `--batch_size` | 64 | 批次大小 |
| `--learning_rate` | 0.001 | 学习率 |
| `--dropout_prob` | 0.25 | Dropout概率 |
| `--test` | - | 快速测试模式 (5 epochs) |
| `--background` | - | 后台运行 |

---

## 📊 监控训练进度

### 查看实时日志

```bash
# 如果是后台运行
tail -f chinese_bimodal_*.log

# 只看 Dev 指标
tail -f chinese_bimodal_*.log | grep "Dev F1"

# 查看GPU使用
watch -n 1 nvidia-smi
```

### 查看最佳指标

```bash
# 查看 Best Dev 指标
grep "BEST DEV SET METRICS" -A 10 chinese_bimodal_*.log

# 查看 Test 指标
grep "TEST SET METRICS" -A 10 chinese_bimodal_*.log
```

---

## 📁 输出文件

实验完成后会生成以下文件：

```
fusion_model_results_test.json    # Test set 性能指标
fusion_model_results_dev.json     # Dev set 性能指标
best_fusion_model.pth              # 最佳模型权重
```

查看结果：
```bash
# 查看 Test set 结果
cat fusion_model_results_test.json | python -m json.tool

# 提取关键指标
python -c "import json; r=json.load(open('fusion_model_results_test.json')); print(f'AUROC: {r[\"auroc\"]:.4f}, F1: {r[\"f1_score\"]:.4f}, Acc: {r[\"accuracy\"]:.4f}')"
```

---

## 📂 配置文件说明

### 1. `constants_chinese_bimodal.py` 

配置文件，定义数据路径和实验参数：

```python
# 数据路径
FINGER_FEATURES_FILE = ".../processed/features_demography_diagnosis_Nov22_2023.csv"
FACIAL_FEATURES_FILE = ".../processed/facial_dataset.csv"

# 双模态配置
MODEL_SUBSETS = {
    0: ['finger_model_both_hand_fusion_baal', 'facial_expression_smile_best_auroc_baal'],
}
```

### 2. `run_chinese_bimodal_fusion.py`

Python运行脚本，会：
1. 备份原始 `constants.py`
2. 替换为 `constants_chinese_bimodal.py`
3. 运行 `uncertainty_aware_fusion.py`
4. 恢复原始 `constants.py`

### 3. `RUN_CHINESE_BIMODAL.sh`

Shell包装脚本，提供便捷的命令行选项。

---

## ⏱️ 时间估计

| 配置 | 预计时间 |
|------|---------|
| 快速测试 (5 epochs) | 5-10 分钟 |
| 标准训练 (244 epochs) | 2-4 小时 |

---

## 🛑 停止实验

```bash
# 如果是前台运行
Ctrl+C

# 如果是后台运行（查看PID）
ps aux | grep run_chinese_bimodal_fusion
kill <PID>

# 或者直接kill所有相关进程
pkill -f "run_chinese_bimodal_fusion"
```

---

## ✅ 验证配置

运行前验证配置是否正确：

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

# 检查配置文件
python -c "import constants_chinese_bimodal as c; print(f'Finger: {c.FINGER_FEATURES_FILE}'); print(f'Smile: {c.FACIAL_FEATURES_FILE}'); print(f'Models: {c.MODEL_SUBSETS[0]}')"

# 检查数据文件
ls -lh /localdisk2/pliu/park_multitask_fusion-main/data/chinese_synthetic_data/real_chinese_smile_finger/processed/

# 统计数据
echo "Finger rows:"; wc -l /localdisk2/pliu/park_multitask_fusion-main/data/chinese_synthetic_data/real_chinese_smile_finger/processed/features_demography_diagnosis_Nov22_2023.csv
echo "Smile rows:"; wc -l /localdisk2/pliu/park_multitask_fusion-main/data/chinese_synthetic_data/real_chinese_smile_finger/processed/facial_dataset.csv
```

---

## 🎯 完整运行示例

### 示例1: 快速测试

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion
./RUN_CHINESE_BIMODAL.sh --test
```

### 示例2: 标准训练（后台）

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion
./RUN_CHINESE_BIMODAL.sh --background

# 查看日志
tail -f chinese_bimodal_*.log

# 查看GPU
watch -n 1 nvidia-smi
```

### 示例3: 自定义参数

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

python run_chinese_bimodal_fusion.py \
  --seed 123 \
  --num_epochs 100 \
  --batch_size 32 \
  --learning_rate 0.0005
```

---

## 📈 预期结果

根据之前的分析：
- **对应ID数**: 55个参与者同时有 Smile 和 Finger 数据
- **数据量**: 相对较小，可能需要调整超参数
- **基线性能**: 参考单模态模型的性能

---

## 🔍 问题排查

### 问题1: 找不到配置文件

```bash
# 确保在正确目录
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

# 检查文件是否存在
ls -l constants_chinese_bimodal.py run_chinese_bimodal_fusion.py
```

### 问题2: GPU内存不足

```bash
# 减小batch size
./RUN_CHINESE_BIMODAL.sh --batch_size 32
```

### 问题3: 数据文件找不到

```bash
# 检查数据路径
python -c "import constants_chinese_bimodal as c; import os; print('Finger exists:', os.path.exists(c.FINGER_FEATURES_FILE)); print('Smile exists:', os.path.exists(c.FACIAL_FEATURES_FILE))"
```

---

## 🎉 开始实验！

### 最推荐的启动方式

```bash
# 1. 先快速测试 (5 epochs, 约5分钟)
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion
./RUN_CHINESE_BIMODAL.sh --test

# 2. 如果测试成功，再运行完整实验（后台）
./RUN_CHINESE_BIMODAL.sh --background

# 3. 监控进度
tail -f chinese_bimodal_*.log
```

---

## 📞 需要帮助？

如有问题，请检查：
1. 配置文件路径是否正确
2. 数据文件是否存在
3. GPU是否可用 (`nvidia-smi`)
4. 日志文件中的错误信息

祝实验顺利！🚀

