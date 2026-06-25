# ─────────────────────────────────────────────────────────────
# day12_plots.py
# All Day 12 visualisations:
#   1. Volatility term structure slopes over time
#   2. VoV (vol-of-vol) vs actual rolling vol comparison
#   3. Feature importance comparison bar chart (top 15)
#   4. Mutual info vs Pearson scatter (linear vs nonlinear signal)
#   5. RFECV optimal feature count curve
#   6. Correlation heatmap of new features with target
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

TICKERS    = ["SPY", "QQQ", "AAPL"]
TICKER_COL = {"SPY": "#1f77b4", "QQQ": "#ff7f0e", "AAPL": "#2ca02c"}


# ── Plot 1: Term structure slopes over time ──────────────────────
def plot_term_structure_slopes(macro_dfs: dict) -> None:
    fig, axes = plt.subplots(len(TICKERS), 1,
                              figsize=(13, 3.5 * len(TICKERS)))
    fig.suptitle("Volatility Term Structure Slope (5d/21d and 21d/63d)",
                 fontsize=13, fontweight="bold")

    for ax, ticker in zip(axes, TICKERS):
        df = macro_dfs.get(ticker)
        if df is None:
            continue
        color = TICKER_COL[ticker]

        if "ts_slope_5_21" in df.columns:
            ax.plot(df.index, df["ts_slope_5_21"],
                    color=color, linewidth=1.0, alpha=0.8,
                    label="5d / 21d slope")
        if "ts_slope_21_63" in df.columns:
            ax.plot(df.index, df["ts_slope_21_63"],
                    color=color, linewidth=1.0, alpha=0.5,
                    linestyle="--", label="21d / 63d slope")

        ax.axhline(1.0, color="black", linewidth=0.8,
                   linestyle=":", alpha=0.5, label="Flat (=1)")
        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.set_ylabel("Slope Ratio")
        ax.legend(frameon=False, fontsize=8, ncol=3)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day12_term_structure_slopes.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── Plot 2: VoV vs rolling vol ───────────────────────────────────
def plot_vov_vs_vol(macro_dfs: dict, ticker: str = "SPY") -> None:
    df = macro_dfs.get(ticker)
    if df is None:
        return

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    fig.suptitle(f"{ticker} — Vol-of-Vol vs Realized Volatility",
                 fontsize=13, fontweight="bold")
    color = TICKER_COL[ticker]

    axes[0].plot(df.index, df.get("rv_rolling_21d", pd.Series()) * 100,
                 color=color, linewidth=1.2)
    axes[0].set_ylabel("21d Rolling Vol (%)")
    axes[0].grid(True, alpha=0.2, linestyle="--")
    axes[0].spines[["top", "right"]].set_visible(False)

    if "vov_21d" in df.columns:
        axes[1].fill_between(df.index, df["vov_21d"],
                              color=color, alpha=0.4)
        axes[1].plot(df.index, df["vov_21d"],
                     color=color, linewidth=0.8)
    axes[1].set_ylabel("Vol-of-Vol (21d std of rv_1d)")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.2, linestyle="--")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"day12_vov_vs_vol_{ticker}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── Plot 3: Feature importance comparison ───────────────────────
def plot_feature_importance_comparison(selection_results: dict, ticker: str = "SPY") -> None:
    if ticker not in selection_results:
        return
    merged = selection_results[ticker]["merged_df"].head(15)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{ticker} — Feature Importance by Method (Top 15)",
                 fontsize=13, fontweight="bold")

    for ax, (col, label, color) in zip(axes, [
        ("pearson_abs_corr", "Pearson |Corr|",  "#1f77b4"),
        ("mutual_info",      "Mutual Info",      "#ff7f0e"),
        ("xgb_gain",         "XGB Gain",         "#2ca02c"),
    ]):
        sub = merged[["feature", col]].dropna().head(15)
        ax.barh(range(len(sub)), sub[col].values[::-1],
                color=color, alpha=0.8, edgecolor="none")
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["feature"].values[::-1], fontsize=8)
        ax.set_xlabel(label)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2, linestyle="--", axis="x")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"day12_feature_importance_{ticker}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── Plot 4: MI vs Pearson scatter ────────────────────────────────
def plot_mi_vs_pearson(selection_results: dict, ticker: str = "SPY") -> None:
    if ticker not in selection_results:
        return

    res     = selection_results[ticker]
    pearson = res["pearson_df"].set_index("feature")["pearson_abs_corr"]
    mi      = res["mi_df"].set_index("feature")["mutual_info"]
    merged  = pd.concat([pearson, mi], axis=1).dropna()
    merged.columns = ["pearson", "mi"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(merged["pearson"], merged["mi"],
               color=TICKER_COL[ticker], alpha=0.65, s=40,
               edgecolors="none")

    for feat, row in merged.iterrows():
        if row["pearson"] > merged["pearson"].quantile(0.75) or \
           row["mi"] > merged["mi"].quantile(0.75):
            ax.annotate(feat, (row["pearson"], row["mi"]),
                        fontsize=6.5, alpha=0.8,
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Pearson |Correlation| (linear signal)")
    ax.set_ylabel("Mutual Information (nonlinear signal)")
    ax.set_title(f"{ticker} — Linear vs Non-Linear Predictive Signal",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"day12_mi_vs_pearson_{ticker}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── Plot 5: RFECV feature count curve ───────────────────────────
def plot_rfecv_curve(selection_results: dict) -> None:
    fig, axes = plt.subplots(1, len(TICKERS),
                              figsize=(5 * len(TICKERS), 4))
    fig.suptitle("RFECV — CV-RMSE vs Number of Features Selected",
                 fontsize=13, fontweight="bold")

    for ax, ticker in zip(axes, TICKERS):
        if ticker not in selection_results:
            continue
        rfecv = selection_results[ticker]["rfecv_res"]
        scores = rfecv["cv_scores"]
        n_opt  = rfecv["n_selected"]
        color  = TICKER_COL[ticker]

        ax.plot(range(1, len(scores) + 1), scores,
                color=color, linewidth=1.5, marker="o",
                markersize=3, alpha=0.8)
        ax.axvline(n_opt, color="red", linewidth=1.5,
                   linestyle="--",
                   label=f"Optimal = {n_opt} features")
        ax.set_xlabel("Number of features")
        ax.set_ylabel("CV RMSE")
        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day12_rfecv_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── Plot 6: Correlation heatmap of all Day 12 new features ──────
def plot_new_feature_correlation_heatmap(macro_dfs: dict, ticker: str = "SPY") -> None:
    df = macro_dfs.get(ticker)
    if df is None:
        return

    rv_1d = df["log_return"] ** 2 * 252
    target = rv_1d.shift(-1).rename("target_rv")

    new_feature_keywords = [
        "ts_slope", "vov", "park_gk", "realized_skew",
        "realized_kurt", "updown", "jump", "atr", "bb_",
        "rsi", "macd", "vol_ratio", "price_ma", "roll_corr",
        "dispersion", "beta", "qqq_spy", "bull_regime",
        "return_streak",
    ]
    new_cols = [c for c in df.columns
                if any(k in c for k in new_feature_keywords)]

    data = pd.concat([df[new_cols], target], axis=1).dropna()

    if data.empty or len(new_cols) == 0:
        print(f"  ⚠ No new feature columns found for {ticker}")
        return

    corr = data.corr()[["target_rv"]].drop("target_rv").sort_values(
        "target_rv", ascending=False
    )

    fig, ax = plt.subplots(figsize=(4, max(6, len(corr) * 0.3)))
    sns.heatmap(
        corr, ax=ax, annot=True, fmt=".2f",
        cmap="RdYlGn_r", center=0,
        vmin=-0.5, vmax=0.5,
        linewidths=0.3,
        cbar_kws={"label": "Corr with target RV"},
    )
    ax.set_title(f"{ticker} — New Feature Correlations with Target RV",
                 fontsize=10, fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)

    plt.tight_layout()
    out = os.path.join(OUT_DIR,
                        f"day12_new_feature_corr_heatmap_{ticker}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── Main ────────────────────────────────────────────────────────
def run_all_plots(macro_dfs: dict, selection_results: dict) -> None:
    print(f"\n{'='*55}")
    print("  DAY 12 — Generating Plots")
    print(f"{'='*55}")

    plot_term_structure_slopes(macro_dfs)

    for ticker in TICKERS:
        plot_vov_vs_vol(macro_dfs, ticker=ticker)
        plot_feature_importance_comparison(selection_results, ticker=ticker)
        plot_mi_vs_pearson(selection_results, ticker=ticker)
        plot_new_feature_correlation_heatmap(macro_dfs, ticker=ticker)

    plot_rfecv_curves(selection_results)

    print("\n  All Day 12 plots complete.")


def plot_rfecv_curves(selection_results: dict) -> None:
    """Alias matching naming convention — calls plot_rfecv_curve."""
    plot_rfecv_curve(selection_results)