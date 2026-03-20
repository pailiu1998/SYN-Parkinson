# Scenario 6 Batch Experiments with Multiple Seeds

这个文档说明如何运行场景6的批量实验，使用随机种子0-100，并计算所有指标的置信区间。

## 📋 文件说明

1. **run_scenario_6_batch.sh** - Bash脚本，循环运行101次实验
2. **run_scenario_6_multiple_seeds.py** - Python脚本，包含完整的实验流程和分析
3. **analyze_scenario_6_results.py** - 分析脚本，计算置信区间和统计信息

## 🚀 使用方法

### 方法1: 使用Bash脚本（推荐用于服务器）

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

# 运行批量实验
./run_scenario_6_batch.sh

# 实验完成后，分析结果
python analyze_scenario_6_results.py scenario_6_results_YYYYMMDD_HHMMSS/
```

### 方法2: 使用Python脚本（更灵活）

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

# 运行批量实验并自动分析
python run_scenario_6_multiple_seeds.py
```

### 方法3: 手动运行单个实验

```bash
# 运行单个seed
python uncertainty_aware_fusion_scenario_6.py --seed=0

# 运行多个seed（使用循环）
for seed in {0..100}; do
    python uncertainty_aware_fusion_scenario_6.py --seed=$seed
done

# 分析结果
python analyze_scenario_6_results.py ./
```

## 📊 输出结果

实验结果会保存在 `scenario_6_results_YYYYMMDD_HHMMSS/` 目录中：

```
scenario_6_results_20231215_123456/
├── test_seed_0.json          # 每个seed的测试集结果
├── test_seed_1.json
├── ...
├── test_seed_100.json
├── dev_seed_0.json           # 每个seed的验证集结果
├── dev_seed_1.json
├── ...
├── dev_seed_100.json
├── log_seed_0.txt            # 每个seed的运行日志（如果使用bash脚本）
├── log_seed_1.txt
├── ...
├── summary_statistics.json   # 汇总统计信息
├── test_all_seeds.csv        # 所有测试集指标的汇总
└── dev_all_seeds.csv         # 所有验证集指标的汇总
```

## 📈 统计指标

脚本会计算以下指标的置信区间：

- **accuracy** - 准确率
- **auroc** - ROC曲线下面积
- **f1_score** - F1分数
- **precision** - 精确率
- **recall** - 召回率/敏感度
- **average_precision** - 平均精确率
- **brier_score** - Brier分数
- **weighted_accuracy** - 加权准确率
- **sensitivity** - 敏感度
- **specificity** - 特异度

对于每个指标，会报告：
- **Mean** - 均值
- **95% CI** - 95%置信区间
- **Std** - 标准差
- **Min/Max** - 最小值/最大值
- **N** - 有效样本数

## 💡 示例输出

```
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

## ⚙️ 自定义配置

如果要修改实验参数，可以编辑相应的脚本：

```python
# 修改种子范围
for seed in range(101):  # 改为 range(50) 只运行0-49

# 修改置信度
compute_confidence_interval(data, confidence=0.99)  # 99%置信区间
```

## 🔍 查看特定种子的结果

```python
import json

# 查看seed=42的测试结果
with open('scenario_6_results_YYYYMMDD_HHMMSS/test_seed_42.json', 'r') as f:
    results = json.load(f)
    print(json.dumps(results, indent=2))
```

## ⏱️ 预计运行时间

- 单个seed大约需要: 5-20分钟（取决于数据大小和GPU性能）
- 101个seeds总计: 约8-34小时

建议在后台运行：

```bash
nohup ./run_scenario_6_batch.sh > run_log.txt 2>&1 &

# 查看进度
tail -f run_log.txt
```

## 🐛 故障排查

如果某些seeds失败：

1. 查看日志文件 `log_seed_X.txt`
2. 单独重新运行失败的seed
3. 分析工具会自动跳过缺失的结果文件

## 📝 注意事项

- 确保有足够的磁盘空间（每个结果文件约几KB到几MB）
- 如果使用GPU，确保没有其他进程占用
- 可以使用 `tmux` 或 `screen` 来保持会话


