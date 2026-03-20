# Batch Training Guide: 100 Seeds with Parallel Execution

## 📋 概述

本指南说明如何并行运行 **Scenario 10** 和 **Original Fusion** 模型的 100 个不同 seeds 的批量训练。

### 关键特性

- ✅ **100 seeds**: 每个模型运行 0-99 共 100 个不同的随机种子
- ✅ **并行执行**: 每次并行运行 10 个训练进程
- ✅ **自动结果收集**: 自动整理和保存所有结果
- ✅ **统计分析**: 计算置信区间和对比分析
- ✅ **容错处理**: 自动处理失败的训练任务

---

## 📁 文件说明

### 主要脚本

| 文件 | 说明 |
|------|------|
| `run_scenario_10_100seeds.py` | Scenario 10 批量训练脚本 |
| `run_original_fusion_100seeds.py` | Original Fusion 批量训练脚本 |
| `run_both_100seeds.sh` | 同时启动两个批量训练的 Bash 脚本 |
| `analyze_100seeds_results.py` | 结果分析和统计脚本 |

### 输出目录

```
uncertainty_aware_fusion/
├── scenario_10_results/          # Scenario 10 结果
│   ├── seed_0/
│   │   ├── fusion_model_results_test.json
│   │   └── fusion_model_results_dev.json
│   ├── seed_1/
│   ├── ...
│   ├── seed_99/
│   ├── all_seeds_results.json    # 所有 seeds 的汇总结果
│   ├── run_summary.json          # 运行统计
│   └── statistical_analysis.json # 统计分析
│
├── original_fusion_results/      # Original Fusion 结果
│   ├── seed_0/
│   ├── ...
│   ├── seed_99/
│   ├── all_seeds_results.json
│   ├── run_summary.json
│   └── statistical_analysis.json
│
└── comparison_scenario10_vs_original.json  # 对比分析
```

---

## 🚀 使用方法

### 方法 1: 同时运行两个模型（推荐）

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

# 给脚本添加执行权限
chmod +x run_both_100seeds.sh

# 启动批量训练
./run_both_100seeds.sh
```

**说明**:
- Scenario 10 和 Original Fusion 会**同时**在后台运行
- 每个模型并行运行 10 个进程
- 输出会保存到各自的日志文件

### 方法 2: 单独运行某个模型

#### 只运行 Scenario 10

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

nohup python run_scenario_10_100seeds.py > scenario_10_batch_log.txt 2>&1 &
```

#### 只运行 Original Fusion

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

nohup python run_original_fusion_100seeds.py > original_fusion_batch_log.txt 2>&1 &
```

---

## 📊 监控训练进度

### 查看实时日志

```bash
# Scenario 10
tail -f scenario_10_batch_log.txt

# Original Fusion
tail -f original_fusion_batch_log.txt
```

### 查看运行进程

```bash
# 查看主进程
ps aux | grep 'run_scenario_10_100seeds\|run_original_fusion_100seeds' | grep python

# 查看训练子进程
ps aux | grep 'uncertainty_aware_fusion' | grep python | wc -l
```

### 查看完成进度

```bash
# Scenario 10
ls scenario_10_results/ | grep "seed_" | wc -l

# Original Fusion
ls original_fusion_results/ | grep "seed_" | wc -l
```

---

## 📈 分析结果

### 运行统计分析

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

python analyze_100seeds_results.py
```

**输出**:
1. 每个场景的统计分析（均值、标准差、95% 置信区间）
2. 两个场景的对比分析
3. 详细的 JSON 格式结果

### 查看分析结果

```bash
# 查看 Scenario 10 统计
cat scenario_10_results/statistical_analysis.json

# 查看 Original Fusion 统计
cat original_fusion_results/statistical_analysis.json

# 查看对比分析
cat comparison_scenario10_vs_original.json
```

---

## 🎯 配置参数

### 修改并行数量

如果想调整并行运行的进程数，编辑对应的 Python 脚本：

```python
# 在 run_scenario_10_100seeds.py 或 run_original_fusion_100seeds.py 中
PARALLEL_JOBS = 10  # 改成你想要的数量，如 5 或 15
```

### 修改 Seeds 范围

```python
NUM_SEEDS = 100     # 总共运行的 seeds 数量
START_SEED = 0      # 起始 seed
```

### 修改训练超参数

```python
# 在 run_experiment 函数中修改命令
cmd = [
    "python", SCRIPT_NAME,
    "--seed", str(seed),
    "--num_epochs", "244",        # 训练轮数
    "--batch_size", "64",         # 批次大小
    "--learning_rate", "0.001",   # 学习率
    "--dropout_prob", "0.25"      # Dropout 概率
]
```

### 修改超时时间

```python
TIMEOUT = 7200  # 2小时，单位：秒
```

---

## 🛑 停止训练

### 停止所有批量训练

```bash
# 查找主进程 PID
ps aux | grep 'run_.*_100seeds.py' | grep python

# 杀死进程（替换 <PID> 为实际的进程 ID）
kill <PID>

# 或使用 pkill
pkill -f "run_scenario_10_100seeds.py"
pkill -f "run_original_fusion_100seeds.py"
```

### 停止所有训练子进程

```bash
# 停止所有 uncertainty_aware_fusion 进程
pkill -f "uncertainty_aware_fusion_scenario_10.py"
pkill -f "uncertainty_aware_fusion.py"
```

---

## 📊 预期输出格式

### run_summary.json

```json
{
  "total_seeds": 100,
  "successful": 98,
  "failed": 1,
  "timeout": 1,
  "errors": 0,
  "total_time_seconds": 25234.56,
  "results": [
    {"seed": 0, "status": "success", "time": 245.3},
    {"seed": 1, "status": "success", "time": 251.8},
    ...
  ]
}
```

### statistical_analysis.json

```json
{
  "test": {
    "f1": {
      "mean": 0.8234,
      "std": 0.0156,
      "se": 0.0016,
      "ci_lower": 0.8203,
      "ci_upper": 0.8265,
      "n": 98
    },
    "auroc": {
      "mean": 0.8567,
      ...
    }
  },
  "dev": {...}
}
```

### comparison_scenario10_vs_original.json

```json
{
  "scenario1": "Scenario 10",
  "scenario2": "Original",
  "test_metrics": {
    "f1": {
      "scenario1_mean": 0.8234,
      "scenario1_ci": [0.8203, 0.8265],
      "scenario2_mean": 0.8156,
      "scenario2_ci": [0.8125, 0.8187],
      "difference": -0.0078
    },
    ...
  }
}
```

---

## ⏱️ 预估时间

### 单个 Seed 训练时间

- **平均**: 3-5 分钟（244 epochs）
- **最长**: 10-15 分钟（取决于 GPU 负载）

### 批量训练总时间

**假设**:
- 100 seeds
- 每个 seed 平均 4 分钟
- 并行 10 个进程

**计算**:
```
总时间 ≈ (100 seeds × 4 min) / 10 parallel = 40 分钟
```

**实际**:
- 考虑启动开销和GPU调度，预计 **45-60 分钟**

---

## 🔍 故障排查

### 问题 1: 某些 seeds 失败

**查看失败原因**:
```bash
cat scenario_10_results/run_summary.json | grep "failed"
```

**解决方法**:
1. 检查日志文件中的错误信息
2. 手动重新运行失败的 seed:
   ```bash
   python uncertainty_aware_fusion_scenario_10.py --seed <失败的seed>
   ```

### 问题 2: GPU 内存不足

**症状**: 进程被杀死或 CUDA out of memory

**解决方法**:
1. 减少并行数量:
   ```python
   PARALLEL_JOBS = 5  # 从 10 改为 5
   ```

2. 减小 batch size:
   ```python
   "--batch_size", "32"  # 从 64 改为 32
   ```

### 问题 3: 训练超时

**解决方法**:
1. 增加超时时间:
   ```python
   TIMEOUT = 10800  # 3小时
   ```

2. 或减少 epochs:
   ```python
   "--num_epochs", "150"  # 从 244 改为 150
   ```

### 问题 4: 结果文件缺失

**症状**: `all_seeds_results.json` 不完整

**解决方法**:
1. 检查各个 seed 目录:
   ```bash
   for i in {0..99}; do
     if [ ! -f "scenario_10_results/seed_$i/fusion_model_results_test.json" ]; then
       echo "Missing: seed_$i"
     fi
   done
   ```

2. 重新运行缺失的 seeds

---

## 💡 最佳实践

### 1. 分阶段运行

如果不确定配置是否正确，先用少量 seeds 测试：

```python
# 修改为只运行 10 个 seeds
NUM_SEEDS = 10
```

### 2. 错峰运行

如果 GPU 资源紧张，考虑：
- 减少并行数量（`PARALLEL_JOBS = 5`）
- 或分两批运行（先运行 Scenario 10，完成后再运行 Original）

### 3. 定期检查

```bash
# 每30分钟检查一次进度
watch -n 1800 'ls scenario_10_results/ | grep seed_ | wc -l'
```

### 4. 备份结果

```bash
# 训练完成后立即备份
tar -czf results_backup_$(date +%Y%m%d).tar.gz \
  scenario_10_results/ original_fusion_results/
```

---

## 📚 相关文档

- [Scenario 10 详细说明](./SCENARIO_10_README.md)
- [单模态训练指南](../../unimodal_models/UNIMODAL_TRAINING_GUIDE.md)
- [Scenarios 4-6 对比](./SCENARIOS_4_5_6_COMPARISON.md)

---

## 🎉 完整工作流程

### Step 1: 准备

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

# 给脚本添加执行权限
chmod +x run_both_100seeds.sh
```

### Step 2: 启动训练

```bash
./run_both_100seeds.sh
```

### Step 3: 监控进度

```bash
# 实时查看日志
tail -f scenario_10_batch_log.txt

# 或在另一个终端查看 original fusion
tail -f original_fusion_batch_log.txt
```

### Step 4: 等待完成

预计 45-60 分钟后完成。

### Step 5: 分析结果

```bash
python analyze_100seeds_results.py
```

### Step 6: 查看分析

```bash
# 查看对比结果
cat comparison_scenario10_vs_original.json

# 或查看详细统计
cat scenario_10_results/statistical_analysis.json
cat original_fusion_results/statistical_analysis.json
```

### Step 7: 备份结果

```bash
tar -czf batch_100seeds_results.tar.gz \
  scenario_10_results/ \
  original_fusion_results/ \
  scenario_10_batch_log.txt \
  original_fusion_batch_log.txt \
  comparison_scenario10_vs_original.json
```

---

**创建时间**: 2024-12-22  
**任务类型**: Batch Training with Statistical Analysis  
**状态**: ✅ Ready to use


