# ─────────────────────────────────────────────────────────────
# day14_plots.py
# Day 14 visualisations:
#   1. HMM regime assignments over time (colour-coded)
#   2. Transition matrix heatmap
#   3. Emission distributions per regime (overlaid histograms)
#   4. Per-regime RMSE comparison bar chart
#   5. Regime-conditional forecast vs actual (SPY)
#   6. Regime posterior probabilities over time
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
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUT_DIR  = os.path.join(BASE_DIR, "output")

TICKERS      = ["SPY", "QQQ", "AAPL"]
REGIME_COLORS = {0: "#2ca02c", 1: "#ff7f0e", 2: "#d62728"}
REGIME_NAMES  = {0: "Low", 1: "Medium", 2: "High"}
MODEL_COLORS  = {
    "GlobalHAR" : "#7f7f7f",
    "RegimeHAR" : "#1f77b4",
    "RegimeXGB" : "#9467bd",
}


# ── Plot 1: Regime assignments over time ────────────────────────
def plot_regime_timeline(hmm_results: dict) -> None:
    n = len(TICKERS)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.5*n))
    if n == 1: axes = [axes]
    fig.suptitle("HMM Volatility Regime Assignments Over Time",
                 fontsize=13, fontweight="bold")

    for ax, ticker in zip(axes, TICKERS):
        if ticker not in hmm_results: continue
        reg = hmm_results[ticker]["regime_series"]
        rv_path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
        if os.path.exists(rv_path):
            df = pd.read_csv(rv_path, index_col="Date", parse_dates=True)
            rv = df["rv_rolling_21d"].dropna()
            ax.plot(rv.index, rv*100, color="black",
                    linewidth=0.7, alpha=0.5, label="21d RV")

        for regime_id, color in REGIME_COLORS.items():
            mask = (reg == regime_id)
            ax.fill_between(reg.index, 0, 1,
                            where=mask.reindex(reg.index, fill_value=False),
                            transform=ax.get_xaxis_transform(),
                            alpha=0.25, color=color,
                            label=f"{REGIME_NAMES[regime_id]} vol")

        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.set_ylabel("21d RV (%)")
        ax.legend(frameon=False, fontsize=7, ncol=4)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines[["top","right"]].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day14_regime_timeline.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 2: Transition matrix heatmap ───────────────────────────
def plot_transition_matrices(hmm_results: dict) -> None:
    n = len(TICKERS)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1: axes = [axes]
    fig.suptitle("HMM Regime Transition Matrices",
                 fontsize=13, fontweight="bold")

    labels = ["Low","Med","High"]
    for ax, ticker in zip(axes, TICKERS):
        if ticker not in hmm_results: continue
        res   = hmm_results[ticker]
        model = res["model"]
        order = res["order"]
        trans = model.transmat_[np.ix_(order, order)]
        sns.heatmap(
            pd.DataFrame(trans, index=labels, columns=labels),
            ax=ax, annot=True, fmt=".2f",
            cmap="Blues", vmin=0, vmax=1,
            linewidths=0.5,
            cbar_kws={"label":"Prob"},
        )
        ax.set_title(ticker, fontsize=11, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day14_transition_matrices.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 3: Emission distributions ──────────────────────────────
def plot_emission_distributions(hmm_results: dict,
                                 ticker: str = "SPY") -> None:
    if ticker not in hmm_results: return
    res   = hmm_results[ticker]
    reg   = res["regime_series"]
    rv_path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
    if not os.path.exists(rv_path): return

    df = pd.read_csv(rv_path, index_col="Date", parse_dates=True)
    rv = df["rv_rolling_21d"].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"{ticker} — RV Distribution by HMM Regime",
                 fontsize=12, fontweight="bold")

    for regime_id, color in REGIME_COLORS.items():
        mask = (reg == regime_id).reindex(rv.index, fill_value=False)
        vals = rv[mask].values * 100
        if len(vals) > 5:
            ax.hist(vals, bins=30, density=True,
                    color=color, alpha=0.55, edgecolor="none",
                    label=f"{REGIME_NAMES[regime_id]} (n={len(vals)})")

    ax.set_xlabel("21d Rolling Vol (%)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"day14_emission_distributions_{ticker}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 4: Per-regime RMSE comparison ──────────────────────────
def plot_per_regime_rmse(metrics_csv: str) -> None:
    df = pd.read_csv(metrics_csv)

    regimes = ["Low","Medium","High","All"]
    models  = df["Model"].unique()
    tickers = sorted(df["Ticker"].unique())

    fig, axes = plt.subplots(1, len(tickers),
                              figsize=(5*len(tickers), 5))
    if len(tickers) == 1: axes = [axes]
    fig.suptitle("RMSE × 10⁴ by Regime and Model",
                 fontsize=13, fontweight="bold")

    x = np.arange(len(regimes))
    w = 0.25

    for ax, ticker in zip(axes, tickers):
        sub = df[df["Ticker"] == ticker]
        for i, model in enumerate(models):
            vals = []
            for reg in regimes:
                m = sub[(sub["Model"]==model) & (sub["Regime"]==reg)]
                vals.append(m["RMSE"].values[0]*1e4 if len(m) else np.nan)
            ax.bar(x + i*w, vals, width=w, label=model,
                   color=MODEL_COLORS.get(model,"#888888"),
                   alpha=0.85, edgecolor="white")

        ax.set_xticks(x + w)
        ax.set_xticklabels(regimes, fontsize=9)
        ax.set_ylabel("RMSE × 10⁴")
        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.2, linestyle="--", axis="y")
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day14_per_regime_rmse.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 5: Forecast vs actual with regime background ───────────
def plot_forecast_with_regime(pred_csv: str,
                               ticker: str = "SPY") -> None:
    df  = pd.read_csv(pred_csv)
    sub = df[df["ticker"]==ticker].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 4))
    x = np.arange(len(sub))

    for regime_id, color in REGIME_COLORS.items():
        mask = (sub["regime"]==regime_id)
        ax.fill_between(x, 0, 1,
                        where=mask,
                        transform=ax.get_xaxis_transform(),
                        alpha=0.15, color=color,
                        label=f"{REGIME_NAMES[regime_id]} regime")

    ax.plot(x, sub["actual"]*100, color="black",
            linewidth=0.8, alpha=0.6, label="Actual")
    ax.plot(x, sub["pred_global_har"]*100,
            color=MODEL_COLORS["GlobalHAR"],
            linewidth=1.0, alpha=0.8, label="GlobalHAR")
    ax.plot(x, sub["pred_regime_har"]*100,
            color=MODEL_COLORS["RegimeHAR"],
            linewidth=1.0, alpha=0.8, label="RegimeHAR")
    ax.plot(x, sub["pred_xgb_regime"]*100,
            color=MODEL_COLORS["RegimeXGB"],
            linewidth=1.0, alpha=0.8, label="RegimeXGB")

    ax.set_title(f"{ticker} — Regime-Conditional Forecasts vs Actual",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Annualised RV (%)")
    ax.set_xlabel("Test observation index")
    ax.legend(frameon=False, fontsize=8, ncol=6)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"day14_forecast_regime_{ticker}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 6: Posterior probabilities ─────────────────────────────
def plot_posteriors(hmm_results: dict, ticker: str = "SPY") -> None:
    if ticker not in hmm_results: return
    post = hmm_results[ticker]["posteriors"]

    fig, ax = plt.subplots(figsize=(13, 4))
    for col, color, label in [
        ("post_low",  REGIME_COLORS[0], "P(Low)"),
        ("post_med",  REGIME_COLORS[1], "P(Medium)"),
        ("post_high", REGIME_COLORS[2], "P(High)"),
    ]:
        if col in post.columns:
            ax.plot(post.index, post[col],
                    color=color, linewidth=1.0, alpha=0.85, label=label)

    ax.axhline(0.5, color="black", linewidth=0.7,
               linestyle=":", alpha=0.4)
    ax.set_title(f"{ticker} — HMM Regime Posterior Probabilities",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("P(Regime | Data)")
    ax.set_xlabel("Date")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"day14_posteriors_{ticker}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Main ────────────────────────────────────────────────────────
def run_all_plots(hmm_results: dict) -> None:
    print(f"\n{'='*55}")
    print("  DAY 14 — Generating Plots")
    print(f"{'='*55}")

    metrics_csv = os.path.join(OUT_DIR, "day14_metrics.csv")
    pred_csv    = os.path.join(OUT_DIR, "day14_regime_predictions.csv")

    plot_regime_timeline(hmm_results)
    plot_transition_matrices(hmm_results)
    if os.path.exists(metrics_csv):
        plot_per_regime_rmse(metrics_csv)
    if os.path.exists(pred_csv):
        for ticker in TICKERS:
            plot_forecast_with_regime(pred_csv, ticker=ticker)

    for ticker in TICKERS:
        plot_emission_distributions(hmm_results, ticker=ticker)
        plot_posteriors(hmm_results, ticker=ticker)

    print("\n  All Day 14 plots complete.")