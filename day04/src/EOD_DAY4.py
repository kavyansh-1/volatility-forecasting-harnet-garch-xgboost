import os
import sys
import warnings

import pandas as pd

# Keep Day 4 modules under .vscode importable from project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".vscode"))

from models.evaluation.model_comparison import (
    plot_feature_importance,
    plot_forecast_vs_realized,
    plot_garch_aic_heatmap,
    plot_har_alpha_sensitivity,
    plot_model_comparison_bar,
    plot_rolling_rmse,
)
from models.evaluation.rolling_backset import compute_all_metrics, run_full_backtest
from models.feature_engineering import load_ticker_frames
from models.tune_garch_models import tune_garch_all_tickers
from models.tune_har_model import tune_har_all_tickers
from models.tune_ml_models import tune_xgboost_all_tickers

warnings.filterwarnings("ignore")

TICKERS = ["SPY", "QQQ", "AAPL"]
FIG_DIR = os.path.join("plots", "day4")


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("\n" + "=" * 58)
    print("  VOLATILITY FORECASTING PROJECT - DAY 4")
    print("  HAR vs GARCH vs XGBoost")
    print("=" * 58)

    print("\n[1/5] Loading processed data...")
    dfs = load_ticker_frames(TICKERS)

    print("\n[2/5] Tuning models...")
    har_results = tune_har_all_tickers(dfs)
    garch_results = tune_garch_all_tickers(dfs)
    xgb_results = tune_xgboost_all_tickers(dfs, n_iter=10)

    print("\n[3/5] Running rolling backtests...")
    backtests = {}
    metrics_frames = []

    for ticker in TICKERS:
        har_cfg = {
            "best_alpha": har_results[ticker].get("best_alpha", 0.01),
            "include_gk": har_results[ticker].get("include_gk", True),
        }

        garch_best = garch_results[ticker].get("best_row")
        if garch_best is not None:
            garch_cfg = {
                "best_p": int(garch_best["p"]),
                "best_q": int(garch_best["q"]),
                "best_vol_model": garch_best["vol_model"],
                "best_dist": garch_best["dist"],
            }
        else:
            garch_cfg = {
                "best_p": 1,
                "best_q": 1,
                "best_vol_model": "GARCH",
                "best_dist": "t",
            }

        xgb_cfg = xgb_results[ticker].get("best_params", {})

        bt = run_full_backtest(
            dfs[ticker],
            ticker=ticker,
            har_params=har_cfg,
            garch_params=garch_cfg,
            xgb_params=xgb_cfg,
            initial_train_size=756,
            step_size=21,
        )
        backtests[ticker] = bt

        metrics = compute_all_metrics(bt, ticker)
        if not metrics.empty:
            metrics_frames.append(metrics)

    metrics_df = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    metrics_path = os.path.join("reports", "day4_model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved -> {metrics_path}")

    print("\n[4/5] Generating Day 4 plots...")
    for ticker, bt in backtests.items():
        plot_forecast_vs_realized(bt, ticker, save_dir=FIG_DIR)
        plot_rolling_rmse(bt, ticker, save_dir=FIG_DIR)

        importance = xgb_results[ticker].get("importance")
        if importance is not None and len(importance) > 0:
            plot_feature_importance(importance, ticker, save_dir=FIG_DIR)

    if not metrics_df.empty:
        plot_model_comparison_bar(metrics_df, save_dir=FIG_DIR)

    plot_har_alpha_sensitivity(har_results, save_dir=FIG_DIR)
    plot_garch_aic_heatmap(garch_results, save_dir=FIG_DIR)

    print("\n[5/5] Day 4 complete.")
    print("Saved plots in plots/day4 and summary in reports/day4_model_metrics.csv")


if __name__ == "__main__":
    main()
