# Synthetic Data Validation - Experiment Summary

## 🎯 实验目标

验证**合成模态数据的有效性**，通过系统性比较不同模态组合的融合模型性能，证明：
1. 合成数据能否有效补充真实数据（相比双模态基线）
2. 合成数据质量能否接近真实数据（相比三模态金标准）

## 📊 实验设计

### 7组实验对比

| 组别 | 实验名称 | 模态组合 | 类型 | 目的 |
|-----|---------|---------|------|------|
| 1 | 2mod_smile_finger | Smile + Finger | 双模态基线 | 建立基线性能 |
| 2 | 2mod_smile_speech | Smile + Speech | 双模态基线 | 建立基线性能 |
| 3 | 2mod_finger_speech | Finger + Speech | 双模态基线 | 建立基线性能 |
| 4 | 3mod_smile_finger_synth_speech | Smile + Finger + **合成Speech** | 两真+一合成 | 验证合成效果 |
| 5 | 3mod_smile_synth_finger_speech | Smile + **合成Finger** + Speech | 两真+一合成 | 验证合成效果 |
| 6 | 3mod_synth_smile_finger_speech | **合成Smile** + Finger + Speech | 两真+一合成 | 验证合成效果 |
| 7 | 3mod_all_real | Smile + Finger + Speech | 三模态金标准 | 性能上限 |

### 核心假设

**H1: 合成数据增益假设**
```
Performance(两真 + 一合成) > Performance(两真)
```
如果成立 → 合成数据能有效补充缺失模态

**H2: 合成数据质量假设**
```
Performance(两真 + 一合成) ≈ Performance(三真)  (差距 < 5%)
```
如果成立 → 合成数据质量接近真实数据

## 📁 文件结构

```
park_multitask_fusion-main/
├── 📄 SYNTHETIC_VALIDATION_PLAN.md      # 详细实验计划
├── 📄 QUICK_START_EXPERIMENTS.md         # 快速开始指南
├── 📄 EXPERIMENT_SUMMARY.md              # 本文件（概要）
│
├── code/
│   ├── fusion_models/
│   │   ├── synthetic_validation_experiment.py  # 🔧 实验管理器
│   │   ├── synthetic_generator.py              # 🔧 合成生成器（待实现）
│   │   └── uncertainty_aware_fusion/
│   │       ├── uncertainty_aware_fusion.py     # 现有fusion脚本
│   │       ├── constants.py                    # 配置文件
│   │       └── run_experiment.py               # 🔧 命令行包装器（待实现）
│   │
│   ├── unimodal_models/
│   │   ├── finger_tapping/
│   │   │   ├── unimodal_finger_baal.py
│   │   │   └── constants_dragon_pd.py          # ✅ Dragon-PD配置
│   │   ├── facial_expression_smile/
│   │   └── quick_brown_fox/
│   │
│   └── feature_extraction_pipeline/
│       ├── finger_tapping/
│       │   └── feature_extraction.py           # ✅ 已修改支持新格式
│       ├── facial_expression_smile/
│       └── quick_brown_fox/
│
├── data/
│   ├── finger_tapping/
│   │   ├── features_demography_diagnosis_Nov22_2023.csv  # 原始数据
│   │   └── dragon_pd_features.csv                        # ✅ Dragon-PD数据
│   ├── facial_expression_smile/
│   ├── quick_brown_fox/
│   └── synthetic/                              # 🔧 合成数据（待生成）
│       ├── speech_from_smile_finger.csv
│       ├── finger_from_smile_speech.csv
│       └── smile_from_finger_speech.csv
│
├── models/
│   ├── finger_model_both_hand_fusion_baal/     # ✅ 已有模型
│   ├── fox_model_best_auroc_baal/              # ✅ 已有模型
│   ├── facial_expression_smile_best_auroc_baal/  # ✅ 已有模型
│   └── synthetic/                              # 🔧 合成生成器模型（待训练）
│
└── results/
    └── synthetic_validation/                   # 实验结果
        ├── 2mod_smile_finger/
        ├── 3mod_smile_finger_synth_speech/
        ├── ...
        └── summary_YYYYMMDD_HHMMSS.csv
```

## ✅ 已完成

1. **数据准备**
   - [x] Dragon-PD Finger Tapping 特征提取
   - [x] 数据格式转换函数 `convert_to_target_format()`
   - [x] 4条 Dragon-PD 数据转换并集成

2. **代码框架**
   - [x] 实验管理器 `synthetic_validation_experiment.py`
   - [x] 修改 Finger Tapping 提取脚本支持新格式
   - [x] Dragon-PD 训练配置 `constants_dragon_pd.py`

3. **文档**
   - [x] 详细实验计划 `SYNTHETIC_VALIDATION_PLAN.md`
   - [x] 快速开始指南 `QUICK_START_EXPERIMENTS.md`
   - [x] 实验概要 `EXPERIMENT_SUMMARY.md`

## 🔧 待实现

### 高优先级
1. **合成生成器实现** (`synthetic_generator.py`)
   - 神经网络架构：从两个模态预测第三个
   - 训练逻辑：使用配对的真实数据
   - 推理逻辑：生成合成预测

2. **Fusion脚本命令行化** (`run_experiment.py`)
   - 动态配置modality组合
   - 支持synthetic数据路径
   - 自动化实验运行

3. **提取更多特征**
   - Dragon-PD Smile 特征
   - Dragon-PD Speech 特征
   - 扩展到更多参与者

### 中优先级
4. **训练合成生成器**
   - Speech生成器（从Smile+Finger）
   - Finger生成器（从Smile+Speech）
   - Smile生成器（从Finger+Speech）

5. **运行完整实验**
   - 所有7组实验
   - 收集性能指标
   - 统计显著性检验

6. **结果分析和可视化**
   - 性能对比图表
   - 置信区间计算
   - 实验报告生成

## 🚀 快速运行

### 查看所有实验
```bash
cd /localdisk2/pliu/park_multitask_fusion-main
python code/fusion_models/synthetic_validation_experiment.py --list
```

### 运行单个实验（当前为框架，待实现）
```bash
python code/fusion_models/synthetic_validation_experiment.py \
  --experiment 2mod_smile_finger
```

### 运行所有实验
```bash
python code/fusion_models/synthetic_validation_experiment.py \
  --experiment all
```

## 📊 评估指标

### 主要指标
- **AUROC**: ROC曲线下面积（主要）
- **Accuracy**: 分类准确率
- **F1 Score**: 精确率和召回率的调和平均
- **Precision/Recall**: 查准率/查全率
- **Brier Score**: 概率校准质量

### 比较维度
1. **横向对比**: 不同模态组合的性能
2. **纵向对比**: 合成vs真实数据的差距
3. **统计检验**: t检验、置信区间、Bootstrap

## 📈 预期时间线

| 阶段 | 任务 | 估计时间 | 状态 |
|-----|------|---------|------|
| Phase 1 | 数据准备 | 2-3天 | ✅ 部分完成 |
| Phase 2 | 代码实现 | 3-5天 | 🔧 进行中 |
| Phase 3 | 训练生成器 | 2-3天 | ⏳ 待开始 |
| Phase 4 | 运行实验 | 1-2天 | ⏳ 待开始 |
| Phase 5 | 分析结果 | 2-3天 | ⏳ 待开始 |
| **总计** | | **10-16天** | |

## 💡 关键洞察

1. **模块化设计**: 每个实验独立运行，便于并行和调试
2. **可复现性**: 所有配置保存为JSON，结果可追溯
3. **增量验证**: 先跑基线（双模态），再跑合成实验
4. **对照设计**: 金标准（三真实）作为性能上限参考

## 📞 联系方式

如有问题或需要协助，请参考：
- 详细计划：`SYNTHETIC_VALIDATION_PLAN.md`
- 快速指南：`QUICK_START_EXPERIMENTS.md`
- 实验脚本：`code/fusion_models/synthetic_validation_experiment.py`

---

**最后更新**: 2024-12-02
**状态**: 框架完成，待实现核心组件


