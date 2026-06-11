import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".vscode"))

from src.download_data import download_ohlcv
from src.quality_check import run_quality_check
from src.plot_prices import plot_adj_close, plot_normalised

TICKERS = ["SPY", "QQQ", "AAPL"]
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"


def main():
    print("\n" + "-" * 65)
    print("VOLATILITY FORECASTING PROJECT - DAY 1")
    print("\n" + "-" * 65)

    print("\n[1/4] Downloading OHLCV data...")
    dfs = download_ohlcv(TICKERS, START_DATE, END_DATE, save_dir="data/raw")

    print("\n[2/4] Running quality checks...")
    report = run_quality_check(dfs)
    cols = ["Status", "Rows", "Missing cells", "Duplicate_dates", "Bad_OHLC"]
    available = [c for c in cols if c in report.columns]
    print(report[available].to_string())
    os.makedirs("reports", exist_ok=True)
    report.to_csv("reports/day1_quality_report.csv")
    print("  Saved -> reports/day1_quality_report.csv")

    print("\n[3/4] Generating charts...")
    plot_adj_close(dfs, save_dir="plots")
    plot_normalised(dfs, save_dir="plots")


if __name__ == "__main__":
    main()