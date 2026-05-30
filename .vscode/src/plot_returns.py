# src/plot_returns.py
# PURPOSE: Visualise return distributions and rolling volatility.
#
# Day 2 generates four chart types:
#   1. Return histogram with normal overlay  — shows fat tails
#   2. QQ-plot                               — formally tests normality
#   3. Rolling volatility over time          — shows volatility clustering
#   4. Volatility comparison (all estimators)— side-by-side for one asset
#
# These are diagnostic charts. You look at them BEFORE modelling to understand
# what you're dealing with. If returns are fat-tailed (they will be),
# that motivates GARCH over simple OLS. If volatility clusters (it will),
# that confirms you need a memory model.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy import stats


PALETTE = {
    "SPY":  "#1f77b4",
    "QQQ":  "#ff7f0e",
    "AAPL": "#2ca02c",
}


def plot_return_distributions(dfs: dict, save_dir: str = "plots") -> None:
    """
    Chart 1: Return histogram + fitted normal curve for all assets.
    One subplot per asset. Normal curve uses the same mean and std
    as the actual returns — so any departure from the curve is
    evidence of fat tails or skewness.
    """
    os.makedirs(save_dir, exist_ok=True)
    tickers = list(dfs.keys())
    n       = len(tickers)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    fig.suptitle("Log Return Distributions vs Normal",
                 fontsize=14, fontweight="bold")

    for ax, ticker in zip(axes, tickers):
        r     = dfs[ticker]["log_return"].dropna()
        color = PALETTE.get(ticker, "steelblue")

        # Histogram — density=True so it integrates to 1 (comparable to PDF)
        ax.hist(r, bins=80, density=True, color=color,
                alpha=0.55, edgecolor="none", label="Actual")

        # Fitted normal PDF using same mean and std as data
        x   = np.linspace(r.min(), r.max(), 300)
        pdf = stats.norm.pdf(x, loc=r.mean(), scale=r.std())
        ax.plot(x, pdf, color="black", linewidth=1.8,
                linestyle="--", label="Normal fit")

        # Annotations
        ax.set_title(ticker, fontsize=13, fontweight="bold")
        ax.set_xlabel("Log Return", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)

        # Show skew and kurtosis in the top-left corner of each panel
        textstr = (f"Skew = {r.skew():.3f}\n"
                   f"Kurt = {r.kurt():.3f}\n"
                   f"σ    = {r.std()*np.sqrt(252)*100:.1f}% p.a.")
        ax.text(0.03, 0.97, textstr,
                transform=ax.transAxes,
                fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

        ax.legend(fontsize=10, frameon=False)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(save_dir, "return_distributions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_qq(dfs: dict, save_dir: str = "plots") -> None:
    """
    Chart 2: QQ-plot (Quantile-Quantile) for each asset.

    What to read in a QQ-plot:
        - Points on the diagonal = data matches normal distribution exactly
        - S-shaped curve (tails curve away from line) = fat tails
        - Asymmetric S = skewness

    Stock returns almost always show the S-shape — heavier tails than normal.
    This is WHY GARCH uses a Student-t distribution instead of Gaussian.
    You'll see this clearly in the plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    tickers = list(dfs.keys())
    n       = len(tickers)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle("QQ-Plot: Log Returns vs Normal Distribution",
                 fontsize=14, fontweight="bold")

    for ax, ticker in zip(axes, tickers):
        r     = dfs[ticker]["log_return"].dropna()
        color = PALETTE.get(ticker, "steelblue")

        # stats.probplot computes theoretical vs observed quantiles
        (theoretical_q, observed_q), (slope, intercept, r2) = stats.probplot(r)

        # Scatter: observed vs theoretical quantiles
        ax.scatter(theoretical_q, observed_q,
                   color=color, alpha=0.3, s=6, label="Observed quantiles")

        # Perfect-normal reference line
        line_x = np.array([min(theoretical_q), max(theoretical_q)])
        ax.plot(line_x, slope * line_x + intercept,
                color="black", linewidth=1.5, linestyle="--", label="Normal ref")

        ax.set_title(ticker, fontsize=13, fontweight="bold")
        ax.set_xlabel("Theoretical Quantiles", fontsize=11)
        ax.set_ylabel("Sample Quantiles",      fontsize=11)
        ax.text(0.05, 0.93, f"R² = {r2:.4f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        ax.legend(fontsize=9, frameon=False)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(save_dir, "qq_plots.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_rolling_volatility(dfs: dict, save_dir: str = "plots") -> None:
    """
    Chart 3: 21-day rolling volatility over time for all assets.

    What to look for:
        - Volatility clustering: calm periods → suddenly spiky → calm again.
          This is the key stylised fact that GARCH models. If volatility
          were random and unclustered, GARCH wouldn't beat a simple average.
        - 2020 COVID spike should be clearly visible in all three assets.
        - AAPL should generally show higher vol than SPY (single stock vs index).
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))

    for ticker, df in dfs.items():
        col = "rv_rolling_21d"
        if col not in df.columns:
            continue
        ax.plot(df.index, df[col] * 100,         # Convert to percentage
                color=PALETTE.get(ticker, "steelblue"),
                linewidth=1.2, label=ticker, alpha=0.85)

    ax.set_title("21-Day Rolling Historical Volatility (Annualised)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Volatility (% p.a.)", fontsize=11)
    ax.set_xlabel("Date",                fontsize=11)
    ax.legend(fontsize=12, frameon=False)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    plt.tight_layout()
    out = os.path.join(save_dir, "rolling_volatility.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_vol_estimator_comparison(dfs: dict,
                                  ticker: str = "SPY",
                                  save_dir: str = "plots") -> None:
    """
    Chart 4: Compare all volatility estimators for one asset.
    Overlays close-to-close, Parkinson, and Garman-Klass on one chart.

    Why this matters:
        Parkinson and GK should be smoother than close-to-close because
        they use more information per day. If they diverge significantly,
        something interesting happened (gap opens, earnings, etc).
        The fact that all three broadly agree validates your data.
    """
    os.makedirs(save_dir, exist_ok=True)

    df = dfs[ticker]

    # Use 21-day window for all estimators
    cols = {
        "Close-to-Close (21d)" : ("rv_rolling_21d", "#1f77b4"),
        "Parkinson (21d)"      : ("park_vol_21d",   "#ff7f0e"),
        "Garman-Klass (21d)"   : ("gk_vol_21d",     "#2ca02c"),
    }

    fig, ax = plt.subplots(figsize=(14, 5))

    for label, (col, color) in cols.items():
        if col in df.columns:
            ax.plot(df.index, df[col] * 100,
                    label=label, color=color, linewidth=1.3, alpha=0.85)

    ax.set_title(f"{ticker} — Volatility Estimator Comparison (21-Day Window)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Volatility (% p.a.)", fontsize=11)
    ax.set_xlabel("Date",                fontsize=11)
    ax.legend(fontsize=11, frameon=False)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    plt.tight_layout()
    out = os.path.join(save_dir, f"{ticker}_estimator_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")