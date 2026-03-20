#!/bin/bash

# Navigate to the script directory
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

echo "========================================================================"
echo "Starting Scenario 7 (Real Finger, Synthetic Smile + Quick)"
echo "========================================================================"
python uncertainty_aware_fusion_scenario_7.py --seed=42

echo ""
echo "========================================================================"
echo "Starting Scenario 8 (Real Quick, Synthetic Smile + Finger)"
echo "========================================================================"
python uncertainty_aware_fusion_scenario_8.py --seed=42

echo ""
echo "========================================================================"
echo "Starting Scenario 9 (Real Smile, Synthetic Quick + Finger)"
echo "========================================================================"
python uncertainty_aware_fusion_scenario_9.py --seed=42

echo ""
echo "========================================================================"
echo "All scenarios completed!"
echo "========================================================================"


