# 场景6批量实验 - 快速开始

## 🎯 目标
运行场景6的实验，使用随机种子0-100，计算所有指标的95%置信区间。

## 📍 第一步：进入正确目录

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion
```

## 🧪 第二步：测试运行（推荐）

先用2个seed测试，确保一切正常（约10-40分钟）：

```bash
python test_scenario_6.py
```

如果测试成功，你会看到：
```
✓ All tests passed! You can now run the full batch:
  python run_scenario_6_multiple_seeds.py
```

## 🚀 第三步：运行完整实验

### 选项A: 前台运行（可以看到实时输出）

```bash
python run_scenario_6_multiple_seeds.py
```

**优点**: 可以实时看到每个epoch的Dev F1, AUROC, Balanced Accuracy
**缺点**: 需要保持终端连接（约8-34小时）

### 选项B: 后台运行（推荐）

使用 tmux（推荐）:
```bash
# 创建新session
tmux new -s scenario6

# 运行实验
python run_scenario_6_multiple_seeds.py

# 按 Ctrl+B 然后按 D detach
# 稍后重新连接: tmux attach -t scenario6
```

或使用 nohup:
```bash
nohup python run_scenario_6_multiple_seeds.py > scenario6.log 2>&1 &

# 查看进度
tail -f scenario6.log
```

## 📊 第四步：查看结果

结果会自动保存在 `scenario_6_results_YYYYMMDD_HHMMSS/` 目录。

运行完成后会显示：

```
==============================================================================
TEST SET RESULTS (95% Confidence Intervals)
==============================================================================
accuracy                 : 0.8523 ± 0.0234  [95% CI: 0.8477 - 0.8569]
auroc                    : 0.9145 ± 0.0189  [95% CI: 0.9107 - 0.9183]
f1_score                 : 0.8342 ± 0.0267  [95% CI: 0.8289 - 0.8395]
...

Results saved to: scenario_6_results_20231215_123456
```

## 📁 输出文件

- `summary_statistics.json` - 所有指标的均值和置信区间
- `test_all_seeds.csv` - 所有测试集结果的表格
- `dev_all_seeds.csv` - 所有验证集结果的表格
- `test_seed_X.json` - 每个seed的详细结果

## ⏱️ 时间估计

- 测试运行 (2 seeds): 10-40 分钟
- 完整运行 (101 seeds): 8-34 小时

## 🔍 监控进度

```bash
# 查看GPU使用
watch -n 1 nvidia-smi

# 查看日志（如果使用nohup）
tail -f scenario6.log

# 查看已完成的seeds数量
ls scenario_6_results_*/test_seed_*.json | wc -l
```

## ❓ 问题排查

### 某个seed失败了？
- 脚本会自动跳过并继续
- 最后会列出所有失败的seeds
- 可以单独重新运行: `python uncertainty_aware_fusion_scenario_6.py --seed=X`

### GPU内存不足？
- 减小batch_size: `python uncertainty_aware_fusion_scenario_6.py --seed=X --batch_size=32`

### SSH断开连接？
- 使用 tmux 或 screen 来保持会话

## 📚 更多信息

- 详细文档: `HOW_TO_RUN_SCENARIO_6.md`
- 完整说明: `README_scenario_6_batch.md`

## 🎉 完成！

实验完成后，你将得到：
- ✅ 101个随机种子的实验结果
- ✅ 所有指标的95%置信区间
- ✅ 完整的统计分析


