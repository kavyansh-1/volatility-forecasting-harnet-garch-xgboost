# day2_run.py
# PURPOSE: Day 2 master script — returns, volatility, distributions, saving.
# Run from the project root: python day2_run.py

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np

from src.compute_returns    import compute_all_returns, describe_returns
from src.compute_volatility import add_all_volatility, describe_volatility
from src.plot_returns       import (plot_return_distributions,
                                    plot_qq,
                                    plot_rolling_volatility,
                                    plot_vol_estimator_comparison)

TICKERS = ["SPY", "QQQ", "AAPL"]


def main():
    print("\n" + "=" * 58)
    print("  VOLATILITY FORECASTING PROJECT — DAY 2")
    print("  Returns · Volatility · Distributions")
    print("=" * 58)

    # ── Step 1: Load raw data ─────────────────────────────────────────────────
    print("\n[1/5]  Loading raw CSVs...")
    dfs_raw = {}
    for t in TICKERS:
        path = f"data/raw/{t}_daily.csv"
        dfs_raw[t] = pd.read_csv(path, index_col="Date", parse_dates=True)
        print(f"  ✓  {t}: {len(dfs_raw[t]):,} rows")

    # ── Step 2: Compute returns ───────────────────────────────────────────────
    print("\n[2/5]  Computing log returns...")
    dfs = compute_all_returns(dfs_raw)

    ret_summary = describe_returns(dfs)
    print("\n  Return Summary Statistics:")
    print(ret_summary.to_string())

    # ── Step 3: Compute volatility ────────────────────────────────────────────
    print("\n[3/5]  Computing volatility estimators...")
    dfs = {t: add_all_volatility(df) for t, df in dfs.items()}

    vol_summary = describe_volatility(dfs)
    print("\n  Volatility Summary:")
    print(vol_summary.to_string())

    # ── Step 4: Save processed data ───────────────────────────────────────────
    print("\n[4/5]  Saving processed data...")
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports",        exist_ok=True)

    for ticker, df in dfs.items():
        path = f"data/processed/{ticker}_processed.csv"
        df.to_csv(path)
        print(f"  ✓  Saved → {path}  ({len(df):,} rows, {len(df.columns)} cols)")

    ret_summary.to_csv("reports/day2_return_stats.csv")
    vol_summary.to_csv("reports/day2_volatility_stats.csv")
    print("  ✓  Saved → reports/day2_return_stats.csv")
    print("  ✓  Saved → reports/day2_volatility_stats.csv")

    # ── Step 5: Generate plots ────────────────────────────────────────────────
    print("\n[5/5]  Generating charts...")
    plot_return_distributions(dfs)
    plot_qq(dfs)
    plot_rolling_volatility(dfs)
    plot_vol_estimator_comparison(dfs, ticker="SPY")

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  Day 2 complete.")
    print("─" * 58)
    print("  data/processed/   — enriched DataFrames (returns + vol)")
    print("  plots/            — 4 new diagnostic charts")
    print("  reports/          — return and volatility stats tables")
    print("─" * 58)
    print("\n  Tomorrow (Day 3): stationarity tests, ACF/PACF,")
    print("  autocorrelation in returns and squared returns.")
    print("=" * 58)


if __name__ == "__main__":
    main()