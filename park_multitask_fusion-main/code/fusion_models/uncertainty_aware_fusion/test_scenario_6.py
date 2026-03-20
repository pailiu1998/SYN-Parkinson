#!/usr/bin/env python
"""
Test script: Run scenario 6 with just 2 seeds to verify everything works
"""
import os
import json
import subprocess
import sys

def test_single_seed(seed):
    """Test running a single seed"""
    print(f"\n{'='*70}")
    print(f"Testing seed {seed}")
    print(f"{'='*70}\n")
    
    cmd = [
        sys.executable,
        "uncertainty_aware_fusion_scenario_6.py",
        f"--seed={seed}"
    ]
    
    try:
        result = subprocess.run(cmd, timeout=3600)
        
        if result.returncode == 0:
            print(f"\n✓ Seed {seed} completed successfully!")
            
            # Check if result files exist
            if os.path.exists("fusion_model_results_test.json"):
                print(f"✓ Test results file created")
                with open("fusion_model_results_test.json", 'r') as f:
                    data = json.load(f)
                    print(f"  - Predictions: {len(data.get('prediction', []))} samples")
                    print(f"  - Labels: {len(data.get('label', []))} samples")
            
            if os.path.exists("fusion_model_results_dev.json"):
                print(f"✓ Dev results file created")
                with open("fusion_model_results_dev.json", 'r') as f:
                    data = json.load(f)
                    print(f"  - Predictions: {len(data.get('prediction', []))} samples")
                    print(f"  - Labels: {len(data.get('label', []))} samples")
            
            return True
        else:
            print(f"\n✗ Seed {seed} failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n✗ Seed {seed} timed out (>1 hour)")
        return False
    except Exception as e:
        print(f"\n✗ Seed {seed} failed with exception: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("Testing Scenario 6 with 2 seeds")
    print("="*70)
    print("\nThis will run seeds 0 and 1 to verify everything works correctly.")
    print("If successful, you can run the full batch with:")
    print("  python run_scenario_6_multiple_seeds.py")
    print("\n" + "="*70 + "\n")
    
    # Test with 2 seeds
    seeds = [0, 1]
    results = {}
    
    for seed in seeds:
        success = test_single_seed(seed)
        results[seed] = success
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    successful = sum(1 for v in results.values() if v)
    print(f"Successful: {successful}/{len(seeds)}")
    print(f"Failed: {len(seeds) - successful}/{len(seeds)}")
    
    if successful == len(seeds):
        print("\n✓ All tests passed! You can now run the full batch:")
        print("  python run_scenario_6_multiple_seeds.py")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        print("Failed seeds:", [seed for seed, success in results.items() if not success])
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()


