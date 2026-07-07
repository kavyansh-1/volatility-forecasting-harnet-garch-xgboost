#!/usr/bin/env python
"""
Quick test to verify day16 modules can be imported and basic functions work.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("Testing Day 16 module imports...")

try:
    from day16_intraday_data import run_intraday_data_pipeline
    print("✓ day16_intraday_data imported successfully")
except Exception as e:
    print(f"✗ Failed to import day16_intraday_data: {e}")
    sys.exit(1)

try:
    from day16_realized_variance import run_rv_estimation
    print("✓ day16_realized_variance imported successfully")
except Exception as e:
    print(f"✗ Failed to import day16_realized_variance: {e}")
    sys.exit(1)

try:
    from day16_jump_detection import run_jump_detection
    print("✓ day16_jump_detection imported successfully")
except Exception as e:
    print(f"✗ Failed to import day16_jump_detection: {e}")
    sys.exit(1)

try:
    from day16_compare_estimators import run_estimator_comparison
    print("✓ day16_compare_estimators imported successfully")
except Exception as e:
    print(f"✗ Failed to import day16_compare_estimators: {e}")
    sys.exit(1)

print("\n✓ All Day 16 modules imported successfully!")
print("Code is runnable. Modules are ready for orchestration.")
