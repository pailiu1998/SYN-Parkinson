# 中文数据双模态融合实验结果

## 实验信息

**实验日期**: 2026-01-20  
**实验名称**: Chinese Bimodal Fusion (Smile + Finger)  
**运行时间**: 04:49:28  
**训练轮数**: 244 epochs  
**训练耗时**: ~15秒 (16.77 it/s)

---

## 数据集信息

- **Smile 数据**: 56个中文参与者
- **Finger 数据**: 62个中文参与者  
- **对应ID**: 55个参与者
- **测试集大小**: 49个样本
- **Dev集大小**: 约330个样本

---

## 最佳 Dev Set 指标 (Best on Dev)

| 指标 | 数值 |
|------|------|
| **Dev Accuracy** | 0.81364 (81.36%) |
| **Dev AUROC** | 0.89139 (89.14%) |
| **Dev Balanced Accuracy** | 0.8005 (80.05%) |
| **Dev F1 Score** | 0.75152 (75.15%) |
| **Dev Loss** | 0.41169 |
| **Dev ECE** | 0.06206 |

---

## Test Set 性能指标 (Final Performance)

### 主要指标

| 指标 | 数值 |
|------|------|
| **Test Accuracy** | 0.6327 (63.27%) |
| **Test AUROC** | 0.8000 (80.00%) |
| **Test F1 Score** | 0.5714 (57.14%) |
| **Average Precision** | 0.6836 (68.36%) |
| **Weighted Accuracy** | 0.6540 (65.40%) |
| **Coverage** | 0.9245 (92.45%) |

### 详细指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **Sensitivity (TPR/Recall)** | 0.8000 (80.00%) | 真阳性率 |
| **Specificity (TNR)** | 0.5588 (55.88%) | 真阴性率 |
| **Precision (PPV)** | 0.4444 (44.44%) | 阳性预测值 |
| **NPV** | 0.8636 (86.36%) | 阴性预测值 |
| **FPR** | 0.4412 (44.12%) | 假阳性率 |

### 校准指标

| 指标 | 数值 |
|------|------|
| **Brier Score (BS)** | 0.2169 |
| **ECE (Expected Calibration Error)** | 0.1119 |
| **Loss** | 0.6199 |

---

## 混淆矩阵 (Confusion Matrix)

|  | 预测负类 | 预测正类 |
|---|---------|---------|
| **实际负类** | TN = 19 | FP = 15 |
| **实际正类** | FN = 3 | TP = 12 |

### 解读

- **True Negatives (TN)**: 19 - 正确预测为健康的样本
- **False Positives (FP)**: 15 - 误判为PD的健康样本
- **False Negatives (FN)**: 3 - 漏诊的PD患者
- **True Positives (TP)**: 12 - 正确识别的PD患者

**总测试样本**: 49 (19 + 15 + 3 + 12)

---

## 训练曲线

训练损失趋势（40个epoch）:
```
train_loss: █▆▃▄▄▃▁▂▂▁▄▃▂▄▃▃▂▂▂▂▃▃▁▃▂▄▂▄▄▃▃▁▃▃▄▂▁▄▃▂
```

最终训练损失: 0.09235

---

## 关键发现

### ✅ 优势

1. **高灵敏度 (80%)**: 能够识别大部分PD患者，漏诊率低
2. **高NPV (86.36%)**: 预测为健康的样本大概率确实健康
3. **良好的AUROC (80%)**: 模型具有较好的判别能力
4. **高覆盖率 (92.45%)**: 模型对大部分样本都有较高的置信度

### ⚠️ 需要改进

1. **特异性较低 (55.88%)**: 健康人群中有44%被误判为PD
2. **精确度较低 (44.44%)**: 预测为PD的样本中，超过一半是误报
3. **FPR较高 (44.12%)**: 假阳性率偏高
4. **Dev-Test性能差距**: Dev AUROC=89.14% vs Test AUROC=80%，存在一定的泛化gap

### 📊 性能对比

| 指标 | Dev Set | Test Set | 差距 |
|------|---------|----------|------|
| AUROC | 89.14% | 80.00% | -9.14% |
| Accuracy | 81.36% | 63.27% | -18.09% |
| F1 Score | 75.15% | 57.14% | -18.01% |
| Balanced Accuracy | 80.05% | 67.94%* | -12.11% |

*Balanced Accuracy = (Sensitivity + Specificity) / 2 = (80% + 55.88%) / 2 = 67.94%

---

## 建议与改进方向

### 1. 数据层面
- **扩充测试集**: 当前测试集仅49个样本，可能导致指标不稳定
- **类别平衡**: 检查测试集中PD vs 健康的比例
- **数据增强**: 考虑对少数类进行数据增强

### 2. 模型层面
- **调整阈值**: 当前使用0.5作为分类阈值，可以根据实际需求调整
  - 降低阈值 → 提高灵敏度（更少漏诊）
  - 提高阈值 → 提高特异性（更少误诊）
- **集成学习**: 使用多个随机种子训练多个模型，进行投票或平均
- **正则化**: Dev-Test gap较大，可能需要更强的正则化

### 3. 融合策略
- **权重调整**: 当前是uncertainty-aware fusion，可以尝试调整fusion权重
- **添加第三模态**: 如果有speech数据，可以考虑三模态融合
- **不确定性校准**: ECE=0.1119，可以通过temperature scaling等方法改进校准

---

## WandB 实验链接

- **Run**: https://wandb.ai/roc-hci/park_final_experiments/runs/bj0jf5k6
- **Project**: https://wandb.ai/roc-hci/park_final_experiments
- **Run Name**: atomic-leaf-1759

---

## 输出文件

✓ `fusion_model_results_test.json` - 测试集详细结果  
✓ `fusion_model_results_dev.json` - 验证集详细结果  
✓ `best_fusion_model.pth` - 最佳模型权重  

---

## 实验配置

```python
EXPERIMENT_NAME = "chinese_bimodal_smile_finger"
RANDOM_SEED = 42
NUM_EPOCHS = 244
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DROPOUT_PROB = 0.25
```

**数据路径**:
- Finger: `data/chinese_synthetic_data/real_chinese_smile_finger/processed/features_demography_diagnosis_Nov22_2023.csv`
- Smile: `data/chinese_synthetic_data/real_chinese_smile_finger/processed/facial_dataset.csv`

**模型**:
- `finger_model_both_hand_fusion_baal`
- `facial_expression_smile_best_auroc_baal`

---

## 结论

本次实验在中文数据上进行了 **Smile + Finger 双模态融合**，取得了以下结果：

1. **AUROC = 80%**: 表明模型具有较好的判别能力
2. **灵敏度 = 80%**: 能够发现大部分PD患者，适合筛查场景
3. **特异性 = 55.88%**: 假阳性率较高，可能需要二次确认

相比Dev set，Test set性能有所下降（AUROC从89.14%降至80%），这可能是由于：
- 测试集样本量较小（49个样本）
- 中文数据与训练数据的分布差异
- 模型在英文数据上训练，在中文数据上的泛化性能有限

**总体评价**: 模型在中文数据上展现了一定的泛化能力，AUROC=80%是可接受的性能，但仍有较大改进空间。

---

**实验日期**: 2026-01-20 04:49  
**记录人**: Automated System  
**状态**: ✅ 实验成功完成

