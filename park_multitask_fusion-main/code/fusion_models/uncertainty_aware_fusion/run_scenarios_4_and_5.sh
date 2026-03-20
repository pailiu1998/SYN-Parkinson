#!/bin/bash
# Run scenario 4 and 5 with seeds 0-100 using nohup

echo "=========================================="
echo "Starting Scenario 4 and 5 experiments"
echo "=========================================="
echo ""
echo "This script will run both scenarios in the background using nohup."
echo "Each scenario will run with seeds 0-100."
echo ""
echo "Logs will be saved to:"
echo "  - scenario_4_run.log"
echo "  - scenario_5_run.log"
echo ""
echo "Results will be saved to:"
echo "  - scenario_4_results_YYYYMMDD_HHMMSS/"
echo "  - scenario_5_results_YYYYMMDD_HHMMSS/"
echo ""
echo "=========================================="
echo ""

# Check if Python script exists
if [ ! -f "run_scenario_4_multiple_seeds.py" ]; then
    echo "Error: run_scenario_4_multiple_seeds.py not found!"
    exit 1
fi

if [ ! -f "run_scenario_5_multiple_seeds.py" ]; then
    echo "Error: run_scenario_5_multiple_seeds.py not found!"
    exit 1
fi

# Start Scenario 4
echo "Starting Scenario 4 in background..."
nohup python run_scenario_4_multiple_seeds.py > scenario_4_run.log 2>&1 &
SCENARIO_4_PID=$!
echo "  - Scenario 4 PID: $SCENARIO_4_PID"
echo "  - Log file: scenario_4_run.log"
echo ""

# Wait a few seconds before starting scenario 5
sleep 5

# Start Scenario 5
echo "Starting Scenario 5 in background..."
nohup python run_scenario_5_multiple_seeds.py > scenario_5_run.log 2>&1 &
SCENARIO_5_PID=$!
echo "  - Scenario 5 PID: $SCENARIO_5_PID"
echo "  - Log file: scenario_5_run.log"
echo ""

echo "=========================================="
echo "Both scenarios are now running in background!"
echo "=========================================="
echo ""
echo "Process IDs:"
echo "  - Scenario 4: $SCENARIO_4_PID"
echo "  - Scenario 5: $SCENARIO_5_PID"
echo ""
echo "To monitor progress:"
echo "  tail -f scenario_4_run.log"
echo "  tail -f scenario_5_run.log"
echo ""
echo "To check if still running:"
echo "  ps aux | grep run_scenario"
echo ""
echo "To stop a scenario:"
echo "  kill $SCENARIO_4_PID  # Stop Scenario 4"
echo "  kill $SCENARIO_5_PID  # Stop Scenario 5"
echo ""
echo "Estimated completion time: 8-34 hours per scenario"
echo "=========================================="


