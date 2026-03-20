#!/bin/bash
# 中文数据双模态融合实验一键运行脚本
# Chinese Bimodal Fusion Quick Start Script

echo "=========================================================================="
echo "🇨🇳 中文数据双模态融合实验 (Smile + Finger)"
echo "   Chinese Bimodal Fusion Experiment"
echo "=========================================================================="
echo ""

# 进入正确目录
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

# 默认配置
SEED=42
EPOCHS=244
BATCH_SIZE=64
LR=0.001
DROPOUT=0.25

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)
            SEED="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --test)
            echo "🧪 Test mode: Running 5 epochs only"
            EPOCHS=5
            shift
            ;;
        --background)
            BACKGROUND=1
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --seed SEED          Random seed (default: 42)"
            echo "  --epochs EPOCHS      Number of epochs (default: 244)"
            echo "  --batch_size SIZE    Batch size (default: 64)"
            echo "  --test               Quick test with 5 epochs"
            echo "  --background         Run in background with nohup"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # 标准运行"
            echo "  $0 --test                    # 快速测试 (5 epochs)"
            echo "  $0 --background              # 后台运行"
            echo "  $0 --epochs 100 --seed 123   # 自定义参数"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# 显示配置
echo "📊 Configuration:"
echo "   - Random Seed: $SEED"
echo "   - Epochs: $EPOCHS"
echo "   - Batch Size: $BATCH_SIZE"
echo "   - Learning Rate: $LR"
echo "   - Dropout: $DROPOUT"
echo ""

# 检查文件是否存在
if [ ! -f "run_chinese_bimodal_fusion.py" ]; then
    echo "❌ Error: run_chinese_bimodal_fusion.py not found!"
    exit 1
fi

if [ ! -f "constants_chinese_bimodal.py" ]; then
    echo "❌ Error: constants_chinese_bimodal.py not found!"
    exit 1
fi

# 运行实验
if [ "$BACKGROUND" = "1" ]; then
    LOG_FILE="chinese_bimodal_$(date +%Y%m%d_%H%M%S).log"
    echo "🚀 Starting experiment in background..."
    echo "   Log file: $LOG_FILE"
    echo ""
    nohup python run_chinese_bimodal_fusion.py \
        --seed $SEED \
        --num_epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LR \
        --dropout_prob $DROPOUT \
        > $LOG_FILE 2>&1 &
    
    PID=$!
    echo "✓ Experiment started with PID: $PID"
    echo ""
    echo "📊 Monitor progress:"
    echo "   tail -f $LOG_FILE"
    echo ""
    echo "🛑 Stop experiment:"
    echo "   kill $PID"
else
    echo "🚀 Starting experiment..."
    echo ""
    python run_chinese_bimodal_fusion.py \
        --seed $SEED \
        --num_epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LR \
        --dropout_prob $DROPOUT
fi

echo ""
echo "=========================================================================="
echo "✅ Script completed!"
echo "=========================================================================="

