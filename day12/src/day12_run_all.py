# ─────────────────────────────────────────────────────────────
# day12_run_all.py
# Master orchestrator for Day 12.
# Runs: vol surface → technical → macro → feature selection → plots.
# Passes enriched DataFrames between modules to avoid re-reading CSVs.
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
from day12_vol_surface_features import run_vol_surface_features
from day12_technical_features   import run_technical_features
from day12_macro_features       import run_macro_features
from day12_feature_selection    import run_feature_selection
from day12_plots                import run_all_plots


def main():
    t0 = time.time()
    print(f"\n{'='*55}")
    print("  DAY 12 — Feature Engineering Deep Dive")
    print("  Vol Surface · Technical · Macro · Selection")
    print(f"{'='*55}")

    # Module 1: Volatility surface proxy features
    print("\n[1/5] Volatility surface features...")
    vol_dfs = run_vol_surface_features()

    # Module 2: Technical indicator features
    print("\n[2/5] Technical indicator features...")
    tech_dfs = run_technical_features(vol_dfs=vol_dfs)

    # Module 3: Macro / regime features
    print("\n[3/5] Macro and regime features...")
    macro_dfs = run_macro_features(tech_dfs=tech_dfs)

    # Count total features per ticker
    for ticker, df in macro_dfs.items():
        n_numeric = sum(1 for c in df.columns
                        if df[c].dtype != object
                        and c not in ["Open","High","Low","Close","Volume"])
        print(f"  {ticker}: {n_numeric} total numeric features in DataFrame")

    # Module 4: Feature selection
    print("\n[4/5] Feature selection (Pearson + MI + XGB + RFECV)...")
    selection_results = run_feature_selection(macro_dfs)

    # Module 5: Plots
    print("\n[5/5] Generating plots...")
    run_all_plots(macro_dfs, selection_results)

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print("  DAY 12 COMPLETE")
    print(f"{'='*55}")
    print(f"  Elapsed : {elapsed:.1f}s")
    print(f"\n  Output files:")
    for f in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"    {f:<55} {sz:>8,} bytes")


if __name__ == "__main__":
    main()