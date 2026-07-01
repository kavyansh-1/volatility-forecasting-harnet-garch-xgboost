# ─────────────────────────────────────────────────────────────
# day13_run_all.py
# Master orchestrator for Day 13.
# Trains all 3 attention architectures for all tickers,
# evaluates metrics, analyses attention patterns, plots everything.
# ─────────────────────────────────────────────────────────────

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from day13_train    import run_training
from day13_evaluate import run_evaluation
from day13_plots    import run_all_plots


def main():
    t0 = time.time()
    print(f"\n{'='*55}")
    print("  DAY 13 — Attention & Transformer Forecasting")
    print(f"{'='*55}")

    # Step 1: Train all architectures
    print("\n[1/3] Training attention models...")
    all_results = run_training()

    # Step 2: Evaluate + attention pattern analysis
    print("\n[2/3] Evaluation and attention analysis...")
    metrics_df, attn_df = run_evaluation(all_results)

    # Step 3: All plots
    print("\n[3/3] Generating plots...")
    run_all_plots(all_results, attn_df)

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print("  DAY 13 COMPLETE")
    print(f"{'='*55}")
    print(f"  Elapsed : {elapsed:.1f}s")
    print(f"\n  Output files:")
    for f in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"    {f:<55} {sz:>8,} bytes")


if __name__ == "__main__":
    main()