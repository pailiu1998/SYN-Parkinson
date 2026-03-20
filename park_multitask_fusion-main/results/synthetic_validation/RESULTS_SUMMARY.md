# Synthetic Validation Experiments - Results Summary

## 已完成实验

| Experiment | Modalities | Type | AUROC | Accuracy | F1 Score | Status |
|-----------|-----------|------|-------|----------|----------|---------|
| Exp 7 | Smile + Finger + Speech | 3-real (Gold) | **0.831** | 75.6% | 0.586 | ✅ Done |
| Exp 1 | Smile + Finger | 2-real | **0.810** | 74.0% | 0.629 | ✅ Done |
| Exp 2 | Smile + Speech | 2-real | - | - | - | ⏳ Next |
| Exp 3 | Finger + Speech | 2-real | - | - | - | ⏳ Pending |
| Exp 4 | Smile + Finger + Synth Speech | 2-real + 1-synth | - | - | - | 🔧 Need synth |
| Exp 5 | Smile + Synth Finger + Speech | 2-real + 1-synth | - | - | - | 🔧 Need synth |
| Exp 6 | Synth Smile + Finger + Speech | 2-real + 1-synth | - | - | - | 🔧 Need synth |

## 初步观察

1. **三模态 vs 双模态（Smile+Finger）**:
   - 三模态 AUROC: 0.831
   - 双模态 AUROC: 0.810
   - **差距: 0.021 (2.5%)** - 添加Speech模态有小幅提升

2. **下一步**: 
   - 运行 Exp2 (Smile + Speech)
   - 运行 Exp3 (Finger + Speech)
   - 建立完整的双模态基线

## Timestamp
生成时间: $(date)
