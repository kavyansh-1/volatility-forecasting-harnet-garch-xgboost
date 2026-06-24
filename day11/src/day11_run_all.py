# ─────────────────────────────────────────────────────────────
# day11_run_all.py
# Master orchestrator for Day 11.
# Runs realized covariance, DCC correlation, portfolio VaR/CVaR,
# and all plots in sequence, passing results between modules
# to avoid redundant recomputation.
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
from day11_realized_covariance import run_realized_covariance
from day11_dcc_correlation     import run_dcc_analysis
from day11_portfolio_var       import run_portfolio_var_analysis
from day11_plots                import run_all_plots


def main():
    t0 = time.time()
    print(f"\n{'='*55}")
    print("  DAY 11 — Portfolio Risk: Covariance, DCC, VaR")
    print(f"{'='*55}")

    # Step 1: Realized covariance (rolling + EWMA)
    print("\n[1/4] Realized covariance estimation...")
    cov_results = run_realized_covariance()

    # Step 2: DCC correlation model
    print("\n[2/4] Dynamic Conditional Correlation (DCC)...")
    dcc_results = run_dcc_analysis()

    # Step 3: Portfolio VaR/CVaR (CCC vs DCC)
    print("\n[3/4] Portfolio VaR/CVaR backtest...")
    port_results = run_portfolio_var_analysis(
        cov_results=cov_results, dcc_results=dcc_results
    )

    # Step 4: Plots
    print("\n[4/4] Generating plots...")
    run_all_plots(cov_results, dcc_results, port_results)

    elapsed = time.time() - t0

    print(f"\n{'='*55}")
    print("  DAY 11 COMPLETE")
    print(f"{'='*55}")
    print(f"  Elapsed : {elapsed:.1f}s")
    print(f"\n  Output files:")
    for f in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"    {f:<50} {sz:>8,} bytes")


if __name__ == "__main__":
    main()