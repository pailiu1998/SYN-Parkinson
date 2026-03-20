# 运行 Scenario 4 和 5 的批量实验

## 🎯 目标

同时运行Scenario 4和5，每个场景使用随机种子0-100，并计算所有指标的95%置信区间。

## 📊 场景说明

- **Scenario 4**: 真实 Finger + Quick，合成 Smile
- **Scenario 5**: 真实 Smile + Finger，合成 Quick

## 🚀 快速开始

### 第1步：进入正确目录

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion
```

### 第2步：一键启动两个场景

```bash
chmod +x run_scenarios_4_and_5.sh
./run_scenarios_4_and_5.sh
```

这会自动：
- ✅ 在后台启动 Scenario 4（使用nohup）
- ✅ 在后台启动 Scenario 5（使用nohup）
- ✅ 将日志分别保存到 `scenario_4_run.log` 和 `scenario_5_run.log`
- ✅ 显示两个进程的PID

### 输出示例

```bash
==========================================
Both scenarios are now running in background!
==========================================

Process IDs:
  - Scenario 4: 12345
  - Scenario 5: 12346

To monitor progress:
  tail -f scenario_4_run.log
  tail -f scenario_5_run.log
```

## 📝 监控进度

### 查看实时日志

```bash
# 查看 Scenario 4 的进度
tail -f scenario_4_run.log

# 查看 Scenario 5 的进度
tail -f scenario_5_run.log

# 同时查看两个日志（分屏）
tmux
# 在tmux中按 Ctrl+B 然后按 " 来分屏
# 上面窗口: tail -f scenario_4_run.log
# 下面窗口: tail -f scenario_5_run.log
```

### 检查进程状态

```bash
# 查看进程是否还在运行
ps aux | grep run_scenario

# 查看GPU使用情况
watch -n 1 nvidia-smi

# 查看已完成的seeds数量
ls scenario_4_results_*/test_seed_*.json | wc -l
ls scenario_5_results_*/test_seed_*.json | wc -l
```

### 查看当前进度

```bash
# Scenario 4进度
echo "Scenario 4 completed seeds:"
ls scenario_4_results_*/test_seed_*.json 2>/dev/null | wc -l

# Scenario 5进度
echo "Scenario 5 completed seeds:"
ls scenario_5_results_*/test_seed_*.json 2>/dev/null | wc -l
```

## 🛑 停止实验

如果需要停止某个场景：

```bash
# 查找PID
ps aux | grep run_scenario

# 停止 Scenario 4
kill <PID_of_scenario_4>

# 停止 Scenario 5
kill <PID_of_scenario_5>

# 或者停止所有
pkill -f "run_scenario_[45]"
```

## 📂 输出文件

### Scenario 4
```
scenario_4_run.log                          # 运行日志
scenario_4_results_YYYYMMDD_HHMMSS/        # 结果目录
├── test_seed_0.json
├── test_seed_1.json
├── ...
├── dev_seed_0.json
├── dev_seed_1.json
├── ...
├── summary_statistics.json
├── test_all_seeds.csv
└── dev_all_seeds.csv
```

### Scenario 5
```
scenario_5_run.log                          # 运行日志
scenario_5_results_YYYYMMDD_HHMMSS/        # 结果目录
├── test_seed_0.json
├── test_seed_1.json
├── ...
├── dev_seed_0.json
├── dev_seed_1.json
├── ...
├── summary_statistics.json
├── test_all_seeds.csv
└── dev_all_seeds.csv
```

## ⏱️ 时间估计

- 每个scenario: 8-34小时（取决于硬件）
- 两个scenario可以并行运行（如果有足够GPU）

## 🔍 查看结果

### 查看实时的Dev指标

在日志文件中，你会看到每个epoch的Dev指标：

```bash
grep "Dev F1" scenario_4_run.log | tail -20
grep "Dev F1" scenario_5_run.log | tail -20
```

### 查看最佳Dev指标

```bash
grep -A 6 "BEST DEV SET METRICS" scenario_4_run.log | tail -20
grep -A 6 "BEST DEV SET METRICS" scenario_5_run.log | tail -20
```

### 查看最终统计结果

```bash
# 在实验完成后
grep -A 20 "TEST SET RESULTS" scenario_4_run.log | tail -30
grep -A 20 "TEST SET RESULTS" scenario_5_run.log | tail -30
```

## 📊 分析结果

实验完成后，可以查看汇总统计：

```bash
# 查看 Scenario 4 的统计
cat scenario_4_results_*/summary_statistics.json | python -m json.tool

# 查看 Scenario 5 的统计
cat scenario_5_results_*/summary_statistics.json | python -m json.tool
```

## 🔧 高级选项

### 只运行单个场景

```bash
# 只运行 Scenario 4
nohup python run_scenario_4_multiple_seeds.py > scenario_4_run.log 2>&1 &

# 只运行 Scenario 5
nohup python run_scenario_5_multiple_seeds.py > scenario_5_run.log 2>&1 &
```

### 修改种子范围

编辑脚本文件，修改这一行：

```python
for seed in range(101):  # 改为 range(10) 只运行 0-9
```

### 使用不同的GPU

如果有多个GPU，可以为每个场景指定不同的GPU：

```bash
# Scenario 4 使用 GPU 0
CUDA_VISIBLE_DEVICES=0 nohup python run_scenario_4_multiple_seeds.py > scenario_4_run.log 2>&1 &

# Scenario 5 使用 GPU 1
CUDA_VISIBLE_DEVICES=1 nohup python run_scenario_5_multiple_seeds.py > scenario_5_run.log 2>&1 &
```

## ❓ 常见问题

### Q: 如何知道实验何时完成？

A: 查看日志文件末尾：
```bash
tail scenario_4_run.log
tail scenario_5_run.log
```

如果看到 "Results saved to:" 和统计结果，说明已完成。

### Q: 实验中断了怎么办？

A: 重新运行脚本，它会从头开始。之前的结果会保存在带时间戳的目录中。

### Q: 可以同时运行Scenario 6吗？

A: 可以，如果GPU内存足够：
```bash
nohup python run_scenario_6_multiple_seeds.py > scenario_6_run.log 2>&1 &
```

### Q: 如何只运行失败的seeds？

A: 查看日志找到失败的seeds，然后单独运行：
```bash
python uncertainty_aware_fusion_scenario_4.py --seed=42
```

## 📞 需要帮助？

查看其他文档：
- `QUICKSTART.md` - 快速开始指南
- `HOW_TO_RUN_SCENARIO_6.md` - 详细运行说明
- `README_scenario_6_batch.md` - 完整文档

## ✅ 检查清单

在运行之前：
- [ ] 已进入正确目录
- [ ] 已给脚本添加执行权限 (`chmod +x run_scenarios_4_and_5.sh`)
- [ ] 有足够的磁盘空间（每个scenario约几GB）
- [ ] GPU可用且内存足够
- [ ] 考虑使用 tmux/screen 以防SSH断开

运行后：
- [ ] 确认两个进程都在运行 (`ps aux | grep run_scenario`)
- [ ] 可以查看日志文件 (`tail -f scenario_4_run.log`)
- [ ] GPU正在使用 (`nvidia-smi`)


