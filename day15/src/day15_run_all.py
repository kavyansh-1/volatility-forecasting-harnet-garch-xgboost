# ─────────────────────────────────────────────────────────────
# day15_run_all.py
# Master orchestrator for Day 15.
# ─────────────────────────────────────────────────────────────

import os, sys, time, warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from day15_online_har     import run_online_comparison
from day15_drift_detection import run_drift_detection
from day15_monitoring      import run_monitoring
from day15_plots           import run_all_plots


def main():
    t0 = time.time()
    print(f"\n{'='*55}")
    print("  DAY 15 — Online Learning · Drift · Monitoring")
    print(f"{'='*55}")

    print("\n[1/4] Online learning comparison...")
    online_results = run_online_comparison()

    print("\n[2/4] Distribution drift detection...")
    drift_results = run_drift_detection()

    print("\n[3/4] Model monitoring...")
    monitors = run_monitoring(online_results, drift_results)

    print("\n[4/4] Generating plots...")
    run_all_plots(online_results, drift_results)

    elapsed = time.time() - t0
    print(f"\n{'='*55}  DAY 15 COMPLETE  {elapsed:.1f}s  {'='*10}")
    for f in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f:<55} {sz:>8,} bytes")


if __name__ == "__main__":
    main()