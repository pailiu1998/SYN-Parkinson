#!/bin/bash

# Run all scenarios for fusion model experiments
# This script runs baseline and 3 scenarios with synthetic data

echo "========================================"
echo "Running Fusion Model Experiments"
echo "========================================"
echo ""

# Baseline: All Real Data
echo ">>> Running BASELINE (All Real Data)..."
python uncertainty_aware_fusion.py
echo ""

# Scenario 1: Real Finger + Synthetic Quick + Synthetic Smile
echo ">>> Running SCENARIO 1 (Real Finger + Synthetic Quick/Smile)..."
python uncertainty_aware_fusion_scenario_1.py
echo ""

# Scenario 2: Synthetic Finger + Synthetic Quick + Real Smile
echo ">>> Running SCENARIO 2 (Synthetic Finger/Quick + Real Smile)..."
python uncertainty_aware_fusion_scenario_2.py
echo ""

# Scenario 3: Synthetic Finger + Real Quick + Synthetic Smile
echo ">>> Running SCENARIO 3 (Synthetic Finger/Smile + Real Quick)..."
python uncertainty_aware_fusion_scenario_3.py
echo ""

echo "========================================"
echo "All experiments completed!"
echo "========================================"





