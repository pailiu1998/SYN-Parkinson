#!/bin/bash

# ============================================================================
# 训练所有三个单模态模型的脚本
# ============================================================================

echo "=========================================="
echo "开始训练所有单模态模型"
echo "=========================================="

SEED=526
NUM_EPOCHS_FINGER=73
NUM_EPOCHS_QUICK=55
NUM_EPOCHS_SMILE=50

# ============================================================================
# 1. Finger Tapping Model
# ============================================================================
echo ""
echo "=========================================="
echo "训练 Finger Tapping 模型..."
echo "=========================================="
cd /localdisk2/pliu/park_multitask_fusion-main/code/unimodal_models/finger_tapping

nohup python unimodal_finger_baal.py \
  --model ShallowANN \
  --dropout_prob 0.13951215957675367 \
  --num_trials 300 \
  --num_buckets 20 \
  --hand both \
  --learning_rate 0.6682837019078968 \
  --random_state 526 \
  --seed $SEED \
  --use_feature_scaling yes \
  --scaling_method StandardScaler \
  --minority_oversample no \
  --batch_size 512 \
  --num_epochs $NUM_EPOCHS_FINGER \
  --drop_correlated no \
  --corr_thr 0.95 \
  --optimizer SGD \
  --momentum 0.8363833208184809 \
  --use_scheduler yes \
  --scheduler step \
  --step_size 22 \
  --gamma 0.6555323541714391 \
  > finger_training_seed_${SEED}.log 2>&1 &

FINGER_PID=$!
echo "Finger Tapping 训练已启动，PID: $FINGER_PID"
echo "日志文件: finger_training_seed_${SEED}.log"

sleep 5

# ============================================================================
# 2. Quick Brown Fox (Audio) Model
# ============================================================================
echo ""
echo "=========================================="
echo "训练 Quick Brown Fox (Audio) 模型..."
echo "=========================================="
cd /localdisk2/pliu/park_multitask_fusion-main/code/unimodal_models/quick_brown_fox

nohup python unimodal_fox_baal.py \
  --model ShallowANN \
  --dropout_prob 0.08349938684379829 \
  --num_trials 5000 \
  --num_buckets 20 \
  --learning_rate 0.9258448866412824 \
  --random_state 526 \
  --seed $SEED \
  --use_feature_scaling no \
  --scaling_method StandardScaler \
  --minority_oversample yes \
  --batch_size 256 \
  --num_epochs $NUM_EPOCHS_QUICK \
  --drop_correlated no \
  --optimizer SGD \
  --momentum 0.49459848722229194 \
  --use_scheduler no \
  > quick_training_seed_${SEED}.log 2>&1 &

QUICK_PID=$!
echo "Quick Brown Fox 训练已启动，PID: $QUICK_PID"
echo "日志文件: quick_training_seed_${SEED}.log"

sleep 5

# ============================================================================
# 3. Facial Expression (Smile) Model
# ============================================================================
echo ""
echo "=========================================="
echo "训练 Facial Expression (Smile) 模型..."
echo "=========================================="
cd /localdisk2/pliu/park_multitask_fusion-main/code/unimodal_models/facial_expression_smile

nohup python unimodal_smile_baal.py \
  --model ShallowANN \
  --dropout_prob 0.1 \
  --num_trials 5000 \
  --num_buckets 20 \
  --learning_rate 0.5 \
  --random_state 526 \
  --seed $SEED \
  --use_feature_scaling yes \
  --scaling_method StandardScaler \
  --minority_oversample no \
  --batch_size 256 \
  --num_epochs $NUM_EPOCHS_SMILE \
  --drop_correlated yes \
  --corr_thr 0.85 \
  --optimizer SGD \
  --momentum 0.9 \
  --use_scheduler yes \
  --scheduler reduce \
  --patience 10 \
  --gamma 0.5 \
  > smile_training_seed_${SEED}.log 2>&1 &

SMILE_PID=$!
echo "Facial Expression 训练已启动，PID: $SMILE_PID"
echo "日志文件: smile_training_seed_${SEED}.log"

echo ""
echo "=========================================="
echo "所有训练任务已启动！"
echo "=========================================="
echo "Finger Tapping PID: $FINGER_PID"
echo "Quick Brown Fox PID: $QUICK_PID"
echo "Facial Expression PID: $SMILE_PID"
echo ""
echo "监控训练进度:"
echo "  tail -f /localdisk2/pliu/park_multitask_fusion-main/code/unimodal_models/finger_tapping/finger_training_seed_${SEED}.log"
echo "  tail -f /localdisk2/pliu/park_multitask_fusion-main/code/unimodal_models/quick_brown_fox/quick_training_seed_${SEED}.log"
echo "  tail -f /localdisk2/pliu/park_multitask_fusion-main/code/unimodal_models/facial_expression_smile/smile_training_seed_${SEED}.log"
echo ""
echo "检查进程状态:"
echo "  ps aux | grep 'unimodal_.*_baal.py' | grep python"
echo ""
echo "=========================================="

