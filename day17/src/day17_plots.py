# ─────────────────────────────────────────────────────────────
# day17_plots.py
# All Day 17 visualisations:
#   1. Return-vol cross-correlation plot (leverage signature)
#   2. News Impact Curves: GJR-GARCH vs EGARCH vs symmetric
#   3. Realized skewness time series
#   4. RS vs next-day vol scatter (does negative RS predict high vol?)
#   5. AHAR model comparison bar chart (RMSE × ticker)
#   6. Asymmetry diagnostic: beta_neg vs beta_pos across lags
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

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR   = os.path.join(BASE_DIR, "output")

TICKERS   = ["SPY", "QQQ", "AAPL"]
TC        = {"SPY": "#1f77b4", "QQQ": "#ff7f0e", "AAPL": "#2ca02c"}
MODEL_COL = {
    "HAR"       : "#7f7f7f",
    "AHAR"      : "#1f77b4",
    "L-HAR"     : "#ff7f0e",
    "Full-AHAR" : "#d62728",
    "GJR-GARCH" : "#9467bd",
    "EGARCH"    : "#17becf",
}


# ── Plot 1: Return-vol cross correlations ───────────────────────
def plot_return_vol_correlation() -> None:
    path = os.path.join(OUT_DIR, "day17_return_vol_corr.csv")
    if not os.path.exists(path):
        return

    df   = pd.read_csv(path)
    lags = sorted(df["Lag"].unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    for ticker in TICKERS:
        sub = df[df["Ticker"] == ticker].sort_values("Lag")
        ax.plot(sub["Lag"], sub["corr_r_rv"],
                color=TC[ticker], linewidth=2.0, marker="o",
                markersize=5, label=f"{ticker} corr(r, future_vol)")

    ax.set_title("Leverage Effect: Corr(r_{t-k}, RV_t) vs Lag k\n"
                 "(negative = down-moves predict higher future vol)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Lag k (days)")
    ax.set_ylabel("Pearson Correlation")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day17_leverage_correlations.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 2: News Impact Curves ───────────────────────────────────
def plot_news_impact_curves(garch_results: dict) -> None:
    n = len(TICKERS)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4.5))
    if n == 1:
        axes = [axes]
    fig.suptitle("News Impact Curves: GJR-GARCH vs EGARCH\n"
                 "(asymmetry = left arm higher than right arm)",
                 fontsize=12, fontweight="bold")

    for ax, ticker in zip(axes, TICKERS):
        if ticker not in garch_results:
            continue
        nic = garch_results[ticker].get("nic_df", pd.DataFrame())
        if nic.empty:
            continue

        for model, color in [("GJR-GARCH", MODEL_COL["GJR-GARCH"]),
                              ("EGARCH",    MODEL_COL["EGARCH"])]:
            sub = nic[nic["model"] == model]
            if sub.empty:
                continue
            ax.plot(sub["eps"], sub["h_next"] * 252 * 100,
                    color=color, linewidth=2.0, label=model)

        # Symmetric reference (parabola centred at 0)
        eps = np.linspace(-5, 5, 100)
        ax.plot(eps, eps**2 * 0.5,
                color="grey", linewidth=1.0, linestyle=":",
                alpha=0.6, label="Symmetric ref")

        ax.axvline(0, color="black", linewidth=0.6, alpha=0.4)
        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.set_xlabel("Innovation ε (std units)")
        ax.set_ylabel("h_{t+1} (annualised, %)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day17_news_impact_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 3: Realized skewness time series ───────────────────────
def plot_realized_skewness() -> None:
    n = len(TICKERS)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.5*n))
    if n == 1:
        axes = [axes]
    fig.suptitle("Realized Skewness (RS) from 5-Minute Returns",
                 fontsize=13, fontweight="bold")

    for ax, ticker in zip(axes, TICKERS):
        path = os.path.join(OUT_DIR,
                             f"day17_realized_moments_{ticker}.csv")
        if not os.path.exists(path):
            continue
        df    = pd.read_csv(path, index_col=0, parse_dates=True)
        color = TC[ticker]

        ax.fill_between(df.index, df["rs"],
                        where=df["rs"] < 0, color="red",
                        alpha=0.35, label="Negative RS (left-skewed)")
        ax.fill_between(df.index, df["rs"],
                        where=df["rs"] > 0, color=color,
                        alpha=0.30, label="Positive RS (right-skewed)")
        ax.plot(df.index, df["rs"].rolling(21).mean(),
                color="black", linewidth=1.0, alpha=0.8,
                label="21-day rolling mean")
        ax.axhline(0, color="black", linewidth=0.7,
                   linestyle="--", alpha=0.4)

        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.set_ylabel("Realized Skewness")
        ax.legend(frameon=False, fontsize=7, ncol=3)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day17_realized_skewness.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 4: RS vs next-day vol scatter ──────────────────────────
def plot_rs_vs_vol(ticker: str = "SPY") -> None:
    path = os.path.join(OUT_DIR, f"day17_realized_moments_{ticker}.csv")
    if not os.path.exists(path):
        return

    df  = pd.read_csv(path, index_col=0, parse_dates=True)
    rs  = df["rs"].dropna()
    rv  = df["rv_ann"].shift(-1).dropna()

    common = rs.index.intersection(rv.index)
    rs_c   = rs.loc[common]
    rv_c   = rv.loc[common]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(rs_c, rv_c * 100, alpha=0.25, s=8,
               c=TC[ticker], edgecolors="none")

    # Trend line
    z   = np.polyfit(rs_c, rv_c, 1)
    x_l = np.linspace(rs_c.quantile(0.01), rs_c.quantile(0.99), 100)
    ax.plot(x_l, np.polyval(z, x_l) * 100,
            color="red", linewidth=2.0, label=f"OLS trend (slope={z[0]:.5f})")

    ax.axvline(0, color="black", linewidth=0.7, linestyle=":", alpha=0.5)
    ax.set_xlabel("Realized Skewness (today)")
    ax.set_ylabel("Next-Day Intraday RV (annualised, %)")
    ax.set_title(f"{ticker} — Realized Skewness vs Next-Day Vol\n"
                 f"(negative slope = leverage effect in realised moments)",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"day17_rs_vs_vol_{ticker}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 5: AHAR model comparison ───────────────────────────────
def plot_ahar_comparison() -> None:
    path = os.path.join(OUT_DIR, "day17_ahar_metrics.csv")
    if not os.path.exists(path):
        return

    df      = pd.read_csv(path)
    tickers = sorted(df["Ticker"].unique())
    models  = ["HAR", "AHAR", "L-HAR", "Full-AHAR"]
    x       = np.arange(len(tickers))
    w       = 0.20

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Asymmetric HAR Models — RMSE and QLIKE Comparison",
                 fontsize=13, fontweight="bold")

    for ax, metric in zip(axes, ["RMSE", "QLIKE"]):
        scale = 1e4 if metric == "RMSE" else 1.0
        label = f"RMSE × 10⁴" if metric == "RMSE" else "QLIKE"

        for i, model in enumerate(models):
            vals = []
            for t in tickers:
                row = df[(df["Ticker"]==t) & (df["Model"]==model)]
                vals.append(
                    row[metric].values[0] * scale if len(row) else np.nan
                )
            color = MODEL_COL.get(model, "#888888")
            ax.bar(x + i*w, vals, width=w, label=model,
                   color=color, alpha=0.85, edgecolor="white")

        ax.set_xticks(x + w * (len(models)-1)/2)
        ax.set_xticklabels(tickers, fontsize=11)
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.2, linestyle="--", axis="y")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day17_ahar_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 6: Beta_neg vs Beta_pos across lags ────────────────────
def plot_asymmetry_coefficients() -> None:
    path = os.path.join(OUT_DIR, "day17_asymmetric_regression.csv")
    if not os.path.exists(path):
        return

    df   = pd.read_csv(path)
    lags = sorted(df["Lag"].unique())

    fig, axes = plt.subplots(1, len(TICKERS),
                              figsize=(5*len(TICKERS), 4.5))
    if len(TICKERS) == 1:
        axes = [axes]
    fig.suptitle("Asymmetric Regression: β_neg vs β_pos by Lag\n"
                 "(|β_neg| > |β_pos| confirms leverage effect)",
                 fontsize=12, fontweight="bold")

    for ax, ticker in zip(axes, TICKERS):
        sub = df[df["Ticker"] == ticker].sort_values("Lag")
        x   = np.arange(len(sub))

        ax.plot(sub["Lag"], sub["beta_pos"], color="green",
                linewidth=1.8, marker="o", markersize=5,
                label="β_pos (from up-moves)")
        ax.plot(sub["Lag"], sub["beta_neg"], color="red",
                linewidth=1.8, marker="s", markersize=5,
                label="β_neg (from down-moves)")
        ax.fill_between(sub["Lag"],
                        sub["beta_neg"], sub["beta_pos"],
                        alpha=0.12, color="purple",
                        label="Asymmetry gap")

        ax.axhline(0, color="black", linewidth=0.7,
                   linestyle=":", alpha=0.5)
        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.set_xlabel("Lag k (days)")
        ax.set_ylabel("Regression coefficient")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day17_asymmetry_coefficients.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Main ────────────────────────────────────────────────────────
def run_all_plots(garch_results: dict) -> None:
    print(f"\n{'='*55}")
    print("  DAY 17 — Generating Plots")
    print(f"{'='*55}")

    plot_return_vol_correlation()
    plot_news_impact_curves(garch_results)
    plot_realized_skewness()
    plot_asymmetry_coefficients()
    plot_ahar_comparison()

    for ticker in TICKERS:
        plot_rs_vs_vol(ticker=ticker)

    print("\n  All Day 17 plots complete.")