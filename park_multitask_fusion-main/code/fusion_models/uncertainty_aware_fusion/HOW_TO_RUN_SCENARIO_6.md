# 如何运行场景6的批量实验

## 🚀 快速开始

### 进入正确的目录
```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion
```

### 方法1: 使用Python脚本（推荐，可以看到实时输出）

```bash
python run_scenario_6_multiple_seeds.py
```

这个脚本会：
- ✅ 自动运行 seed 0-100 的所有实验
- ✅ 显示每个epoch的 Dev F1, AUROC, Balanced Accuracy
- ✅ 自动收集所有结果
- ✅ 计算所有指标的95%置信区间
- ✅ 保存详细统计信息

### 方法2: 使用Bash脚本

```bash
./run_scenario_6_batch.sh
```

### 方法3: 后台运行（推荐用于长时间实验）

```bash
# 使用 nohup 后台运行
nohup python run_scenario_6_multiple_seeds.py > scenario_6_run.log 2>&1 &

# 查看进度
tail -f scenario_6_run.log

# 或者使用 tmux/screen
tmux new -s scenario6
python run_scenario_6_multiple_seeds.py
# 按 Ctrl+B 然后按 D 来detach
```

## 📊 运行时会看到的输出

### 每个Seed的训练过程

```
============================================================
Running experiment with seed: 0
============================================================

Number of patients in the dev and test set: 268, 268
...
Epoch 0: Dev F1: 0.7234, Dev AUROC: 0.8456, Dev Balanced Accuracy: 0.7891, Dev Loss: 0.4532
Epoch 1: Dev F1: 0.7456, Dev AUROC: 0.8567, Dev Balanced Accuracy: 0.8012, Dev Loss: 0.4321
...

======================================================================
BEST DEV SET METRICS
======================================================================
Dev F1 Score:           0.7823
Dev AUROC:              0.8945
Dev Balanced Accuracy:  0.8456
Dev Accuracy:           0.8523
Dev Loss:               0.3876
Dev ECE:                0.0234
======================================================================

============================================================
✓ Seed 0 completed successfully
============================================================
```

### 最终统计结果

```
======================================================================
Computing statistics across 101 successful runs
======================================================================

==============================================================================
TEST SET RESULTS (95% Confidence Intervals)
==============================================================================
accuracy                 : 0.8523 ± 0.0234  [95% CI: 0.8477 - 0.8569]
auroc                    : 0.9145 ± 0.0189  [95% CI: 0.9107 - 0.9183]
f1_score                 : 0.8342 ± 0.0267  [95% CI: 0.8289 - 0.8395]
precision                : 0.8567 ± 0.0301  [95% CI: 0.8507 - 0.8627]
recall                   : 0.8234 ± 0.0312  [95% CI: 0.8172 - 0.8296]
...
```

## 📁 输出文件

结果会保存在 `scenario_6_results_YYYYMMDD_HHMMSS/` 目录中：

```
scenario_6_results_20231215_123456/
├── test_seed_0.json          # 每个seed的测试结果
├── test_seed_1.json
├── ...
├── dev_seed_0.json           # 每个seed的验证结果
├── dev_seed_1.json
├── ...
├── summary_statistics.json   # 汇总统计（包含CI）
├── test_all_seeds.csv        # 所有测试结果的表格
└── dev_all_seeds.csv         # 所有验证结果的表格
```

## ⏱️ 预计时间

- **单个 seed**: 约 5-20 分钟
- **101个 seeds**: 约 8-34 小时

## 🔍 只运行特定的seed

如果你只想测试几个seed：

```bash
# 运行单个seed
python uncertainty_aware_fusion_scenario_6.py --seed=0

# 运行多个特定的seeds
for seed in 0 1 2 3 4; do
    python uncertainty_aware_fusion_scenario_6.py --seed=$seed
done

# 然后分析结果
python analyze_scenario_6_results.py ./
```

## 🛠️ 自定义种子范围

修改 `run_scenario_6_multiple_seeds.py` 中的这一行：

```python
for seed in range(101):  # 改为 range(10) 只运行 0-9
```

## 📈 查看中间结果

在实验运行过程中，你可以查看已经完成的结果：

```bash
# 查看已经生成的结果文件
ls -lh scenario_6_results_*/

# 分析当前已有的结果
python analyze_scenario_6_results.py scenario_6_results_YYYYMMDD_HHMMSS/
```

## 🐛 故障排查

### 如果某个seed失败了

脚本会自动跳过失败的seed并继续，最后会报告：
```
Successful runs: 95/101
Failed runs: 6/101
Failed seeds: [12, 34, 56, 78, 90, 92]
```

你可以单独重新运行失败的seeds：
```bash
python uncertainty_aware_fusion_scenario_6.py --seed=12
```

### 查看GPU使用情况

```bash
watch -n 1 nvidia-smi
```

### 如果内存不足

修改 batch_size（在 uncertainty_aware_fusion_scenario_6.py 中）：
```bash
python uncertainty_aware_fusion_scenario_6.py --seed=0 --batch_size=32
```

## 💡 提示

1. **使用 tmux/screen**: 避免SSH断连导致实验中断
2. **监控磁盘空间**: 确保有足够空间存储结果
3. **定期备份**: 实验运行过程中定期备份 results 目录
4. **GPU监控**: 使用 `nvidia-smi` 监控GPU使用情况

## 📞 需要帮助？

查看详细文档：
```bash
cat README_scenario_6_batch.md
```


