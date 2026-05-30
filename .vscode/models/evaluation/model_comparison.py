# evaluation/model_comparison.py
# PURPOSE: Generate all Day 4 comparison plots and summary tables.
#
# PLOTS GENERATED
# ────────────────────────────────────────────────────────────────────────────
# 1. forecast_vs_realized_{ticker}.png
#    Overlay of model forecasts vs actual RV over the test period.
#    Shows visually whether each model tracks the volatility regime.
#
# 2. rolling_rmse_{ticker}.png
#    90-day rolling RMSE for each model. Shows whether performance
#    is stable over time or degrades in specific market conditions.
#
# 3. model_comparison_bar.png
#    Side-by-side bar chart of RMSE and DirAcc across all models and tickers.
#    The primary summary chart for the report.
#
# 4. hyperparameter_sensitivity_har.png
#    How HAR CV-RMSE changes with Ridge alpha.
#    Visual justification for the chosen regularisation strength.
#
# 5. feature_importance_xgb_{ticker}.png
#    XGBoost feature importances (gain). Shows which RV lags and
#    estimators are most predictive.
#
# 6. garch_grid_heatmap_{ticker}.png
#    AIC heatmap over (p,q) grid for each distribution.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import seaborn as sns

PALETTE = {
    "HAR"    : "#1f77b4",   # blue
    "GARCH"  : "#ff7f0e",   # orange
    "XGBoost": "#2ca02c",   # green
    "Actual" : "#d62728",   # red
}
TICKER_COLORS = {"SPY": "#1f77b4", "QQQ": "#ff7f0e", "AAPL": "#2ca02c"}


def plot_forecast_vs_realized(backtest_df: pd.DataFrame,
                               ticker: str,
                               save_dir: str = "output/figures_day4") -> None:
    """Overlay forecast vs realised RV for all models."""
    os.makedirs(save_dir, exist_ok=True)
    df = backtest_df.dropna(subset=["Actual"]).copy()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df["Actual"] * 100, color=PALETTE["Actual"],
            linewidth=1.0, alpha=0.7, label="Realised RV", zorder=5)

    for col, label in [("HAR_Pred", "HAR"), ("GARCH_Pred", "GARCH"),
                       ("XGB_Pred", "XGBoost")]:
        if col in df.columns and df[col].notna().sum() > 0:
            ax.plot(df.index, df[col] * 100, color=PALETTE[label],
                    linewidth=1.1, alpha=0.8, label=label)

    ax.set_title(f"{ticker} — Forecast vs Realised Volatility (Out-of-Sample)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Annualised RV (%)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(fontsize=10, frameon=False)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    plt.tight_layout()

    out = os.path.join(save_dir, f"forecast_vs_realized_{ticker}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_rolling_rmse(backtest_df: pd.DataFrame,
                      ticker: str,
                      window: int = 90,
                      save_dir: str = "output/figures_day4") -> None:
    """90-day rolling RMSE over the test period for each model."""
    os.makedirs(save_dir, exist_ok=True)
    df = backtest_df.dropna(subset=["Actual"]).copy()

    fig, ax = plt.subplots(figsize=(14, 4))
    for col, label in [("HAR_Pred", "HAR"), ("GARCH_Pred", "GARCH"),
                       ("XGB_Pred", "XGBoost")]:
        if col in df.columns and df[col].notna().sum() > 10:
            sq_err = (df["Actual"] - df[col]) ** 2
            rolling_rmse = np.sqrt(sq_err.rolling(window, min_periods=20).mean())
            ax.plot(df.index, rolling_rmse * 100, color=PALETTE[label],
                    linewidth=1.3, label=label)

    ax.set_title(f"{ticker} — {window}-Day Rolling RMSE (× 100)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Rolling RMSE", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(fontsize=10, frameon=False)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    plt.tight_layout()

    out = os.path.join(save_dir, f"rolling_rmse_{ticker}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_model_comparison_bar(metrics_df: pd.DataFrame,
                               save_dir: str = "output/figures_day4") -> None:
    """Side-by-side bar chart of RMSE and DirAcc for all models × tickers."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Comparison — Out-of-Sample Performance",
                 fontsize=14, fontweight="bold")

    tickers = metrics_df["Ticker"].unique()
    models  = metrics_df["Model"].unique()
    x       = np.arange(len(tickers))
    w       = 0.25

    colors  = [PALETTE[m] for m in models if m in PALETTE]

    for ax, metric, ylabel, fmt in [
        (axes[0], "RMSE",   "RMSE (annualised RV units)", "×10⁴"),
        (axes[1], "DirAcc", "Directional Accuracy (%)",   "%"),
    ]:
        for i, (model, color) in enumerate(zip(models, colors)):
            vals = [
                metrics_df[(metrics_df["Ticker"] == t) &
                           (metrics_df["Model"]  == model)][metric].values[0]
                if len(metrics_df[(metrics_df["Ticker"] == t) &
                                  (metrics_df["Model"]  == model)]) > 0
                else np.nan
                for t in tickers
            ]
            scale = 1e4 if metric == "RMSE" else 1.0
            ax.bar(x + i * w, np.array(vals) * scale,
                   width=w, label=model, color=color, alpha=0.85,
                   edgecolor="white", linewidth=0.5)

        ax.set_xticks(x + w)
        ax.set_xticklabels(tickers, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=10, frameon=False)
        ax.grid(True, alpha=0.25, linestyle="--", axis="y")
        ax.spines[["top", "right"]].set_visible(False)

        if metric == "DirAcc":
            ax.axhline(50, color="red", linestyle="--",
                       linewidth=1.0, alpha=0.6, label="Random baseline (50%)")

    plt.tight_layout()
    out = os.path.join(save_dir, "model_comparison_bar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_feature_importance(importance: pd.Series,
                             ticker: str,
                             save_dir: str = "output/figures_day4") -> None:
    """Horizontal bar chart of XGBoost feature importances (top 12)."""
    os.makedirs(save_dir, exist_ok=True)
    top = importance.head(12)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(top)), top.values[::-1],
            color=PALETTE["XGBoost"], alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance (Gain)", fontsize=11)
    ax.set_title(f"{ticker} — XGBoost Top Feature Importances",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--", axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = os.path.join(save_dir, f"feature_importance_xgb_{ticker}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_har_alpha_sensitivity(har_results: dict,
                                save_dir: str = "output/figures_day4") -> None:
    """How HAR CV-RMSE changes with Ridge alpha (one line per ticker)."""
    os.makedirs(save_dir, exist_ok=True)

    # Rebuild alpha sweep from scratch for the plot
    import warnings; warnings.filterwarnings("ignore")
    from sklearn.linear_model  import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline      import Pipeline
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from models.tune_har_model import build_har_features
    import pandas as pd

    alphas  = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    tscv    = TimeSeriesSplit(n_splits=5)
    fig, ax = plt.subplots(figsize=(9, 5))

    for ticker, res in har_results.items():
        # Load df from processed files
        df = pd.read_csv(
            f"data/processed/{ticker}_processed.csv",
            index_col="Date", parse_dates=True
        )
        X, y = build_har_features(df, include_gk=True)
        n    = len(X)
        X_tr = X.iloc[:n - 500]
        y_tr = y.iloc[:n - 500]

        rmse_list = []
        for a in alphas:
            pipe = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=a))])
            scores = cross_val_score(pipe, X_tr, y_tr, cv=tscv,
                                     scoring="neg_root_mean_squared_error")
            rmse_list.append(-scores.mean())

        ax.plot(alphas, rmse_list,
                marker="o", markersize=5,
                label=ticker, color=TICKER_COLORS[ticker], linewidth=1.5)
        ax.axvline(res["best_alpha"], color=TICKER_COLORS[ticker],
                   linestyle=":", alpha=0.5, linewidth=1.0)

    ax.set_xscale("log")
    ax.set_xlabel("Ridge Alpha (log scale)", fontsize=11)
    ax.set_ylabel("CV RMSE", fontsize=11)
    ax.set_title("HAR-Ridge: CV-RMSE vs Regularisation Strength",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, frameon=False)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = os.path.join(save_dir, "hyperparameter_sensitivity_har.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_garch_aic_heatmap(garch_results: dict,
                            save_dir: str = "output/figures_day4") -> None:
    """AIC heatmap over (p,q) for each ticker × distribution."""
    os.makedirs(save_dir, exist_ok=True)
    tickers = list(garch_results.keys())
    n       = len(tickers)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, ticker in zip(axes, tickers):
        gdf = garch_results[ticker]["grid_df"]
        if gdf is None or len(gdf) == 0:
            ax.set_title(f"{ticker} — no data")
            continue

        # Use only the best distribution per (p,q) pair
        pivot = (gdf.groupby(["p", "q"])["aic"]
                    .min()
                    .unstack("q"))

        sns.heatmap(pivot, ax=ax, annot=True, fmt=".0f",
                    cmap="YlOrRd_r",   # reversed: lower AIC = darker = better
                    linewidths=0.5, cbar_kws={"label": "AIC"})
        ax.set_title(f"{ticker} — GARCH AIC by (p, q)\n(lower = better)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("q (ARCH order)", fontsize=10)
        ax.set_ylabel("p (GARCH order)", fontsize=10)

    plt.tight_layout()
    out = os.path.join(save_dir, "garch_aic_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")