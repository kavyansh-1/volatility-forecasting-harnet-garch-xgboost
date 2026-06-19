# ─────────────────────────────────────────────────────────────
# day09_plots.py
# All Day 9 visualisations:
#   1. Walk-forward RMSE distribution (boxplot per experiment)
#   2. Rolling RMSE over time (expanding vs rolling comparison)
#   3. Prediction intervals with shaded band
#   4. HAR coefficient evolution over time
#   5. XGB feature importance stability heatmap
#   6. Coefficient L2 drift over time
# ─────────────────────────────────────────────────────────────

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR  = os.path.join(BASE_DIR, "output")

PALETTE = {
    "HAR_expanding" : "#1f77b4",
    "HAR_rolling"   : "#aec7e8",
    "XGB_expanding" : "#2ca02c",
    "XGB_rolling"   : "#98df8a",
}
TICKER_COL = {"SPY": "#1f77b4", "QQQ": "#ff7f0e", "AAPL": "#2ca02c"}
TICKERS    = ["SPY", "QQQ", "AAPL"]


# ── Plot 1: RMSE distribution boxplot ──────────────────────────
def plot_rmse_distribution() -> None:
    """
    Boxplot of per-window RMSE for each experiment × ticker.
    Shows median, IQR, and outlier windows.
    A wide box = unstable model. Narrow box = reliable model.
    """
    path = os.path.join(OUT_DIR, "day09_walk_forward_windows.csv")
    if not os.path.exists(path):
        print(f"  ⚠ {path} not found")
        return

    df      = pd.read_csv(path)
    tickers = df["Ticker"].unique()
    n       = len(tickers)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]
    fig.suptitle("Walk-Forward RMSE Distribution per Experiment",
                 fontsize=13, fontweight="bold")

    for ax, ticker in zip(axes, tickers):
        sub  = df[df["Ticker"] == ticker]
        exps = sub["model"].unique()

        data   = [sub[sub["model"] == e]["RMSE"].values * 1e4
                  for e in exps]
        colors = [PALETTE.get(e, "#888888") for e in exps]

        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticks(range(1, len(exps) + 1))
        ax.set_xticklabels(
            [e.replace("_", "\n") for e in exps], fontsize=8
        )
        ax.set_ylabel("RMSE × 10⁴")
        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25, linestyle="--", axis="y")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day09_rmse_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 2: Rolling RMSE over time ─────────────────────────────
def plot_rolling_rmse_over_time() -> None:
    """
    Line chart of per-window RMSE over calendar time.
    Expanding vs Rolling on the same axes.
    Helps identify specific time periods where models break down.
    """
    path = os.path.join(OUT_DIR, "day09_walk_forward_windows.csv")
    if not os.path.exists(path):
        return

    df      = pd.read_csv(path, parse_dates=["test_start"])
    tickers = df["Ticker"].unique()
    n       = len(tickers)

    fig, axes = plt.subplots(n, 1, figsize=(13, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]
    fig.suptitle("Walk-Forward RMSE Over Time\n"
                 "(Expanding vs Rolling Window)",
                 fontsize=13, fontweight="bold")

    for ax, ticker in zip(axes, tickers):
        sub = df[df["Ticker"] == ticker]
        for exp in sub["model"].unique():
            e_sub = sub[sub["model"] == exp].sort_values("test_start")
            ax.plot(
                e_sub["test_start"],
                e_sub["RMSE"] * 1e4,
                color=PALETTE.get(exp, "#888888"),
                linewidth=1.3, alpha=0.85,
                label=exp,
            )

        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.set_ylabel("RMSE × 10⁴")
        ax.legend(frameon=False, fontsize=8, ncol=2)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    axes[-1].set_xlabel("Window test start date")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day09_rmse_over_time.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 3: Prediction intervals ────────────────────────────────
def plot_prediction_intervals() -> None:
    """
    Shaded 90% prediction interval with point forecast and actual.
    One subplot per ticker. Shows last 120 test observations for clarity.
    """
    path = os.path.join(OUT_DIR, "day09_prediction_intervals.csv")
    if not os.path.exists(path):
        return

    df      = pd.read_csv(path, parse_dates=["date"])
    tickers = df["ticker"].unique()
    n       = len(tickers)
    N_SHOW  = 120   # show last 120 test days for readability

    fig, axes = plt.subplots(n, 1, figsize=(13, 4 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle("HAR-Ridge: 90% Bootstrapped Prediction Intervals",
                 fontsize=13, fontweight="bold")

    for ax, ticker in zip(axes, tickers):
        sub   = df[df["ticker"] == ticker].tail(N_SHOW)
        color = TICKER_COL.get(ticker, "steelblue")
        x     = sub["date"]

        # Shaded prediction interval
        ax.fill_between(
            x,
            sub["lower_90"] * 100,
            sub["upper_90"] * 100,
            alpha=0.20, color=color, label="90% PI"
        )
        # Point forecast
        ax.plot(x, sub["point_pred"] * 100,
                color=color, linewidth=1.4,
                label="Point forecast", alpha=0.9)
        # Actual
        ax.plot(x, sub["actual"] * 100,
                color="red", linewidth=0.9,
                alpha=0.7, label="Actual RV")

        # Mark observations outside the PI in orange
        outside = sub[sub["in_interval"] == 0]
        ax.scatter(outside["date"], outside["actual"] * 100,
                   color="orange", s=15, zorder=5,
                   label="Outside PI")

        cov = sub["in_interval"].mean() * 100
        ax.set_title(f"{ticker}  —  Coverage={cov:.1f}%  "
                     f"(target=90%)",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Annualised RV (%)")
        ax.legend(frameon=False, fontsize=8, ncol=4)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day09_prediction_intervals.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 4: HAR coefficient evolution ──────────────────────────
def plot_har_coefficients() -> None:
    """
    Line chart of HAR-Ridge coefficients over walk-forward windows.
    One line per coefficient (rv_lag_1, rv_lag_5, rv_lag_21).
    Stable lines = model is learning genuine structure, not noise.
    """
    n = len(TICKERS)
    fig, axes = plt.subplots(n, 1, figsize=(13, 4 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle("HAR-Ridge Coefficient Evolution Over Time",
                 fontsize=13, fontweight="bold")

    coef_colors = {
        "coef_rv_lag_1" : "#d62728",
        "coef_rv_lag_5" : "#ff7f0e",
        "coef_rv_lag_21": "#1f77b4",
    }
    coef_labels = {
        "coef_rv_lag_1" : "β_daily (rv_lag_1)",
        "coef_rv_lag_5" : "β_weekly (rv_lag_5)",
        "coef_rv_lag_21": "β_monthly (rv_lag_21)",
    }

    for ax, ticker in zip(axes, TICKERS):
        path = os.path.join(OUT_DIR,
                             f"day09_har_coefs_{ticker}.csv")
        if not os.path.exists(path):
            ax.set_title(f"{ticker} — data not found")
            continue

        df = pd.read_csv(path, parse_dates=["window_end"])

        for col, color in coef_colors.items():
            if col in df.columns:
                ax.plot(df["window_end"], df[col],
                        color=color, linewidth=1.5,
                        label=coef_labels[col])

        ax.axhline(0, color="black", linewidth=0.7,
                   linestyle="--", alpha=0.4)
        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.set_ylabel("Coefficient (standardised)")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    axes[-1].set_xlabel("Training window end date")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day09_har_coefs_over_time.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 5: XGB importance stability heatmap ────────────────────
def plot_xgb_stability_heatmap(ticker: str = "SPY") -> None:
    """
    Heatmap: rows = top features, columns = walk-forward windows.
    Colour = normalised feature importance at that window.
    Stable horizontal bands = consistent feature usage.
    Noisy vertical stripes = unstable feature ranking.
    """
    path = os.path.join(OUT_DIR,
                         f"day09_xgb_importance_{ticker}.csv")
    if not os.path.exists(path):
        return

    df   = pd.read_csv(path, parse_dates=["window_end"])
    feat_cols = [c for c in df.columns
                 if c not in ["window_end", "n_train"]]

    # Select top 10 features by mean importance
    top_feats = (df[feat_cols].mean()
                              .sort_values(ascending=False)
                              .head(10).index.tolist())
    matrix = df[top_feats].T   # (features, windows)

    # Normalise each feature across windows to [0,1]
    matrix_norm = matrix.apply(
        lambda r: (r - r.min()) / (r.max() - r.min() + 1e-10),
        axis=1
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(
        matrix_norm,
        ax=ax, cmap="YlOrRd",
        xticklabels=False,
        yticklabels=top_feats,
        cbar_kws={"label": "Normalised importance"},
    )
    ax.set_title(
        f"{ticker} — XGBoost Feature Importance Over Walk-Forward Windows\n"
        f"(each column = one window, normalised row-wise)",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Walk-forward window (time →)")
    ax.set_ylabel("Feature")
    plt.tight_layout()

    out = os.path.join(OUT_DIR,
                        f"day09_xgb_stability_{ticker}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 6: Coefficient drift ───────────────────────────────────
def plot_coefficient_drift() -> None:
    """
    L2 drift of HAR coefficients over time (all tickers on one chart).
    High spikes = parameter instability at that window.
    These often align with market regime changes.
    """
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_title("HAR-Ridge: Coefficient L2 Drift Over Time\n"
                 "(spike = large parameter shift at that window)",
                 fontsize=12, fontweight="bold")

    for ticker in TICKERS:
        path = os.path.join(OUT_DIR,
                             f"day09_har_drift_{ticker}.csv")
        if not os.path.exists(path):
            continue
        df    = pd.read_csv(path, parse_dates=["window_end"])
        color = TICKER_COL.get(ticker, "steelblue")
        ax.plot(df["window_end"], df["l2_drift"],
                color=color, linewidth=1.3,
                alpha=0.85, label=ticker)

    ax.set_ylabel("L2 Drift  ‖Δcoef‖₂")
    ax.set_xlabel("Window end date")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day09_coefficient_drift.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Main ────────────────────────────────────────────────────────
def run_all_plots():
    print(f"\n{'='*55}")
    print("  DAY 9 — Generating Plots")
    print(f"{'='*55}")

    plot_rmse_distribution()
    plot_rolling_rmse_over_time()
    plot_prediction_intervals()
    plot_har_coefficients()
    plot_coefficient_drift()

    for ticker in TICKERS:
        plot_xgb_stability_heatmap(ticker=ticker)

    print("\n  All Day 9 plots complete.")