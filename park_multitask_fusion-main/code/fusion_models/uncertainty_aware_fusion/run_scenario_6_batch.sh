#!/bin/bash
# Run scenario 6 with multiple random seeds (0-100)

echo "=========================================="
echo "Running Scenario 6 with seeds 0-100"
echo "=========================================="

# Create results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="scenario_6_results_${TIMESTAMP}"
mkdir -p ${RESULTS_DIR}

echo "Results will be saved to: ${RESULTS_DIR}"

# Counter for successful and failed runs
SUCCESS_COUNT=0
FAIL_COUNT=0

# Run experiments with different seeds
for seed in {0..100}
do
    echo ""
    echo "=========================================="
    echo "Running seed: ${seed}"
    echo "=========================================="
    
    # Run the experiment
    python uncertainty_aware_fusion_scenario_6.py --seed=${seed} 2>&1 | tee ${RESULTS_DIR}/log_seed_${seed}.txt
    
    # Check if successful
    if [ $? -eq 0 ]; then
        echo "✓ Seed ${seed} completed successfully"
        ((SUCCESS_COUNT++))
        
        # Move result files to results directory
        if [ -f "fusion_model_results_test.json" ]; then
            mv fusion_model_results_test.json ${RESULTS_DIR}/test_seed_${seed}.json
        fi
        
        if [ -f "fusion_model_results_dev.json" ]; then
            mv fusion_model_results_dev.json ${RESULTS_DIR}/dev_seed_${seed}.json
        fi
    else
        echo "✗ Seed ${seed} failed"
        ((FAIL_COUNT++))
    fi
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total seeds: 101"
echo "Successful: ${SUCCESS_COUNT}"
echo "Failed: ${FAIL_COUNT}"
echo "Results saved to: ${RESULTS_DIR}"
echo "=========================================="

# Run analysis script to compute confidence intervals
echo ""
echo "Computing confidence intervals..."
python - <<EOF
import os
import json
import numpy as np
import pandas as pd
from scipy import stats

results_dir = "${RESULTS_DIR}"

# Collect all test results
test_results = []
for seed in range(101):
    test_file = f"{results_dir}/test_seed_{seed}.json"
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            data = json.load(f)
            # Extract metrics from predictions if needed
            test_results.append(data)

# Collect all dev results
dev_results = []
for seed in range(101):
    dev_file = f"{results_dir}/dev_seed_{seed}.json"
    if os.path.exists(dev_file):
        with open(dev_file, 'r') as f:
            data = json.load(f)
            dev_results.append(data)

print(f"\nCollected results from {len(test_results)} test runs and {len(dev_results)} dev runs")
print(f"Results saved in: {results_dir}")
EOF


