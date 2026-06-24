# ─────────────────────────────────────────────────────────────
# day10_run_all.py
# Master orchestrator for Day 10.
# Runs forecast combination, VaR/CVaR backtest, SHAP analysis,
# and all plots in sequence.
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
from day10_forecast_combination  import run_all_combinations
from day10_var_cvar              import run_var_cvar_analysis
from day10_shap_interpretability import run_all_shap_analysis
from day10_plots                 import run_all_plots


def main():
    t0 = time.time()
    print(f"\n{'='*55}")
    print("  DAY 10 — Combination · VaR/CVaR · SHAP")
    print(f"{'='*55}")

    # Step 1: Forecast combination
    print("\n[1/4] Forecast combination...")
    combo_results = run_all_combinations()

    # Step 2: VaR / CVaR backtest
    print("\n[2/4] VaR / CVaR risk analysis...")
    var_summary_df, var_series = run_var_cvar_analysis()

    # Step 3: SHAP interpretability
    print("\n[3/4] SHAP interpretability...")
    shap_results = run_all_shap_analysis()

    # Step 4: Plots
    print("\n[4/4] Generating plots...")
    run_all_plots(var_series=var_series, shap_results=shap_results)

    elapsed = time.time() - t0

    print(f"\n{'='*55}")
    print("  DAY 10 COMPLETE")
    print(f"{'='*55}")
    print(f"  Elapsed : {elapsed:.1f}s")
    print(f"\n  Output files:")
    for f in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"    {f:<55} {sz:>8,} bytes")


if __name__ == "__main__":
    main()