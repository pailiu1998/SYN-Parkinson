# Running Scenarios 7, 8, 9 with Multiple Seeds

This guide explains how to run scenarios 7, 8, and 9 with multiple random seeds (0-100) to calculate confidence intervals for all metrics.

## Overview

- **Scenario 7**: Real Finger, Synthetic Smile + Quick (finger_to_smile_quick)
- **Scenario 8**: Real Quick, Synthetic Smile + Finger (quick_to_smile_finger)
- **Scenario 9**: Real Smile, Synthetic Quick + Finger (smile_to_quick_finger)

Each scenario will be run 101 times (seeds 0-100), and both **test set** and **dev set** metrics with 95% confidence intervals will be calculated.

## Quick Start

To run all three scenarios in the background with separate log files:

```bash
cd /localdisk2/pliu/park_multitask_fusion-main/code/fusion_models/uncertainty_aware_fusion
./run_scenarios_7_8_9.sh
```

This will start three background processes, each running one scenario with 101 different seeds.

## Monitoring Progress

Check the progress of each scenario:

```bash
# For Scenario 7
tail -f scenario_7_run_log.txt

# For Scenario 8
tail -f scenario_8_run_log.txt

# For Scenario 9
tail -f scenario_9_run_log.txt
```

You'll see real-time Dev F1, AUROC, Balanced Accuracy, and Loss for each epoch.

## Running Individual Scenarios

You can also run scenarios individually:

```bash
# Run only Scenario 7
python run_scenario_7_multiple_seeds.py

# Run only Scenario 8
python run_scenario_8_multiple_seeds.py

# Run only Scenario 9
python run_scenario_9_multiple_seeds.py
```

## Output

Each scenario will create a results directory with a timestamp:

```
scenario_7_results_YYYYMMDD_HHMMSS/
├── test_seed_0.json          # Individual test results for each seed
├── test_seed_1.json
├── ...
├── dev_seed_0.json           # Individual dev results for each seed
├── dev_seed_1.json
├── ...
├── test_all_seeds.csv        # All test results in CSV format
├── dev_all_seeds.csv         # All dev results in CSV format
└── summary_statistics.json   # Summary with means and 95% CIs
```

## Metrics Computed

For both **test** and **dev** sets, the following metrics are computed with 95% confidence intervals:

- Accuracy
- AUROC
- F1 Score
- Precision
- Recall
- Average Precision
- Brier Score
- Weighted Accuracy (Balanced Accuracy)
- Sensitivity
- Specificity

## Example Output

```
======================================================================
TEST SET RESULTS
======================================================================
accuracy                 : 0.8842 (95% CI: [0.8721, 0.8963])
auroc                    : 0.9396 (95% CI: [0.9287, 0.9505])
f1_score                 : 0.8281 (95% CI: [0.8142, 0.8420])
weighted_accuracy        : 0.8734 (95% CI: [0.8621, 0.8847])
...

======================================================================
DEV SET RESULTS
======================================================================
accuracy                 : 0.8233 (95% CI: [0.8102, 0.8364])
auroc                    : 0.9225 (95% CI: [0.9108, 0.9342])
f1_score                 : 0.7816 (95% CI: [0.7671, 0.7961])
weighted_accuracy        : 0.8127 (95% CI: [0.8012, 0.8242])
...
```

## Stopping the Experiments

If you need to stop the running experiments:

```bash
# Find the process IDs
ps aux | grep "run_scenario_[789]_multiple_seeds.py"

# Kill specific process
kill <PID>

# Or kill all three
pkill -f "run_scenario_[789]_multiple_seeds.py"
```

## Expected Runtime

- Each single run takes approximately 3-5 minutes
- Total per scenario: ~5-8 hours for 101 runs
- Running all three scenarios in parallel: ~5-8 hours total

## Notes

- The scripts automatically save results after each seed completes
- If a run fails, it will be recorded in the `failed_seeds` list
- Results are saved incrementally, so you can check partial results even while experiments are running
- Each scenario logs to its own file: `scenario_7_run_log.txt`, `scenario_8_run_log.txt`, `scenario_9_run_log.txt`


