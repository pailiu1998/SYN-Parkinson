#!/bin/bash
# Train fusion models for all scenarios

cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

echo "================================================================================"
echo "Training Fusion Models for Chinese Data"
echo "================================================================================"

echo ""
echo "Starting Real Bimodal training (Smile + Finger)..."
python train_real_bimodal.py 2>&1 | tee logs/real_bimodal_training.log
echo "Real Bimodal training completed!"

echo ""
echo "Starting Scenario 1 training (Smile + Synthetic Finger + Synthetic Speech)..."
python train_scenario1.py 2>&1 | tee logs/scenario1_training.log
echo "Scenario 1 training completed!"

echo ""
echo "Starting Scenario 2 training (Smile + Real Finger + Synthetic Speech)..."
python train_scenario2.py 2>&1 | tee logs/scenario2_training.log
echo "Scenario 2 training completed!"

echo ""
echo "Starting Scenario 3 training (Smile + Synthetic Speech)..."
python train_scenario3.py 2>&1 | tee logs/scenario3_training.log
echo "Scenario 3 training completed!"

echo ""
echo "================================================================================"
echo "All trainings completed!"
echo "================================================================================"
echo "Models saved to:"
echo "  - /localdisk2/pliu/park_multitask_fusion-main/models/fusion_real_bimodal/"
echo "  - /localdisk2/pliu/park_multitask_fusion-main/models/fusion_scenario1/"
echo "  - /localdisk2/pliu/park_multitask_fusion-main/models/fusion_scenario2/"
echo "  - /localdisk2/pliu/park_multitask_fusion-main/models/fusion_scenario3/"

