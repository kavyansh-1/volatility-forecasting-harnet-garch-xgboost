"""
EOD_DAY5.py  (place this in your project ROOT — same level as EOD_DAY1.py)
---------------------------------------------------------------------------
Day 5 Task: ACF / PACF Diagnostic Plots
  - ACF and PACF of log returns (all assets)
  - ACF and PACF of squared returns (all assets)
  - Ljung-Box p-value chart (confirms volatility clustering)
  - Clean, consistently-styled PNGs saved to plots/day5/

Run:
    python EOD_DAY5.py
"""

import os
import sys

# ── path fix so Python finds the src/ and models/ modules ────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".vscode"))

from src.plot_acf_pacf_day5 import run_day5

if __name__ == "__main__":
    run_day5(
        tickers  = ["SPY", "QQQ", "AAPL"],
        data_dir = "data/processed",
        save_dir = "plots/day5",
        nlags    = 40,
    )
