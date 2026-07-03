# ─────────────────────────────────────────────────────────────
# day14_run_all.py
# Master orchestrator for Day 14.
# ─────────────────────────────────────────────────────────────

import os, sys, time, warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from day14_hmm_regimes  import run_hmm_all_tickers
from day14_regime_models import run_regime_models
from day14_evaluate     import run_evaluation
from day14_plots        import run_all_plots


def main():
    t0 = time.time()
    print(f"\n{'='*55}")
    print("  DAY 14 — Regime Detection & Conditional Forecasting")
    print(f"{'='*55}")

    print("\n[1/4] HMM regime detection...")
    hmm_results = run_hmm_all_tickers()

    print("\n[2/4] Regime-conditional models...")
    regime_results = run_regime_models(hmm_results)

    print("\n[3/4] Evaluation...")
    metrics_df, dm_df = run_evaluation(regime_results)

    print("\n[4/4] Plots...")
    run_all_plots(hmm_results)

    elapsed = time.time() - t0
    print(f"\n{'='*55}  DAY 14 COMPLETE  {elapsed:.1f}s  {'='*10}")
    for f in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f:<52} {sz:>8,} bytes")


if __name__ == "__main__":
    main()