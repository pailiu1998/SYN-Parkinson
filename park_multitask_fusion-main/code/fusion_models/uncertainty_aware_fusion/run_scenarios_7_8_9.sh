#!/bin/bash

# Navigate to the script directory
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion

echo "Starting Scenario 7 experiments with nohup..."
nohup python run_scenario_7_multiple_seeds.py > scenario_7_run_log.txt 2>&1 &
echo "Scenario 7 started in background. Check scenario_7_run_log.txt for output."
echo "PID for Scenario 7: $!"

echo ""
echo "Starting Scenario 8 experiments with nohup..."
nohup python run_scenario_8_multiple_seeds.py > scenario_8_run_log.txt 2>&1 &
echo "Scenario 8 started in background. Check scenario_8_run_log.txt for output."
echo "PID for Scenario 8: $!"

echo ""
echo "Starting Scenario 9 experiments with nohup..."
nohup python run_scenario_9_multiple_seeds.py > scenario_9_run_log.txt 2>&1 &
echo "Scenario 9 started in background. Check scenario_9_run_log.txt for output."
echo "PID for Scenario 9: $!"

echo ""
echo "To check progress for Scenario 7: tail -f scenario_7_run_log.txt"
echo "To check progress for Scenario 8: tail -f scenario_8_run_log.txt"
echo "To check progress for Scenario 9: tail -f scenario_9_run_log.txt"
echo ""
echo "All three scenarios are now running in the background!"


