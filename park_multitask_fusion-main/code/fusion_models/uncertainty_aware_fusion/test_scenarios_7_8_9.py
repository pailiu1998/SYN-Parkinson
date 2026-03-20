#!/usr/bin/env python
"""
Quick test script to verify scenarios 7, 8, 9 work correctly
Runs each scenario with just 2 seeds (0, 1) for quick validation
"""
import subprocess
import sys

def test_scenario(scenario_num, seeds=[0, 1]):
    """Test a single scenario with given seeds"""
    print(f"\n{'='*70}")
    print(f"Testing Scenario {scenario_num} with seeds: {seeds}")
    print(f"{'='*70}\n")
    
    for seed in seeds:
        cmd = [
            sys.executable,
            f"uncertainty_aware_fusion_scenario_{scenario_num}.py",
            f"--seed={seed}"
        ]
        
        print(f"\nRunning seed {seed}...")
        try:
            result = subprocess.run(cmd, timeout=600)  # 10 min timeout
            if result.returncode == 0:
                print(f"✓ Scenario {scenario_num}, Seed {seed} completed successfully")
            else:
                print(f"✗ Scenario {scenario_num}, Seed {seed} failed with return code: {result.returncode}")
                return False
        except Exception as e:
            print(f"✗ Scenario {scenario_num}, Seed {seed} failed with exception: {e}")
            return False
    
    return True

def main():
    print("="*70)
    print("Quick Test of Scenarios 7, 8, 9")
    print("This will run each scenario with seeds 0 and 1")
    print("="*70)
    
    results = {}
    
    # Test each scenario
    for scenario in [7, 8, 9]:
        success = test_scenario(scenario)
        results[scenario] = success
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for scenario, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"Scenario {scenario}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All scenarios passed! You can now run the full batch.")
        print("Run: ./run_scenarios_7_8_9.sh")
    else:
        print("\n✗ Some scenarios failed. Please check the errors above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())


