# ─────────────────────────────────────────────────────────────
# day15_plots.py
# All Day 15 visualisations:
#   1. Online vs offline prediction comparison
#   2. EWMA coefficient evolution over time
#   3. PSI drift scores over time with threshold bands
#   4. Feature-level PSI heatmap
#   5. Monitoring dashboard: RMSE + drift + alert traffic lights
#   6. Prediction error distribution: online vs offline
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
import matplotlib.patches as mpatches
import seaborn as sns

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR  = os.path.join(BASE_DIR, "output")

TICKERS   = ["SPY", "QQQ", "AAPL"]
TICKER_COL= {"SPY": "#1f77b4", "QQQ": "#ff7f0e", "AAPL": "#2ca02c"}
MODEL_COL = {
    "EWMA_Online" : "#9467bd",
    "SGD_Online"  : "#17becf",
    "Offline_21d" : "#7f7f7f",
}
ALERT_COL = {"GREEN": "#2ca02c", "YELLOW": "#ff7f0e", "RED": "#d62728"}


# ── Plot 1: Online vs offline prediction comparison ──────────────
def plot_online_comparison(online_results: dict,
                            ticker: str = "SPY") -> None:
    if ticker not in online_results: return
    res   = online_results[ticker]
    y     = res["y"].values
    n     = len(y)
    burn  = 252
    x_idx = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle(f"{ticker} — Online vs Offline Forecasting",
                 fontsize=13, fontweight="bold")

    # Top: predictions vs actual
    ax = axes[0]
    ax.plot(x_idx, y * 100, color="red",
            linewidth=0.7, alpha=0.6, label="Actual RV")
    for label, preds, color in [
        ("EWMA Online", res["preds_ewma"],   MODEL_COL["EWMA_Online"]),
        ("SGD Online",  res["preds_sgd"],    MODEL_COL["SGD_Online"]),
        ("Offline 21d", res["preds_offline"],MODEL_COL["Offline_21d"]),
    ]:
        valid = ~np.isnan(preds)
        ax.plot(x_idx[valid], preds[valid] * 100,
                color=color, linewidth=1.0, alpha=0.8, label=label)

    ax.axvline(burn, color="black", linewidth=0.8,
               linestyle="--", alpha=0.5, label="Burn-in end")
    ax.set_ylabel("Annualised RV (%)")
    ax.legend(frameon=False, fontsize=8, ncol=4)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)

    # Bottom: rolling squared error (63-day)
    ax2 = axes[1]
    for label, preds, color in [
        ("EWMA Online", res["preds_ewma"],   MODEL_COL["EWMA_Online"]),
        ("Offline 21d", res["preds_offline"],MODEL_COL["Offline_21d"]),
    ]:
        valid = ~np.isnan(preds)
        sq_err = np.where(valid, (y - preds)**2, np.nan)
        roll_rmse = pd.Series(sq_err).rolling(63, min_periods=10).mean()
        roll_rmse = np.sqrt(roll_rmse)
        ax2.plot(x_idx, roll_rmse * 1e4, color=color,
                 linewidth=1.2, label=label)

    ax2.set_ylabel("63-day Rolling RMSE (×10⁴)")
    ax2.set_xlabel("Observation index")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(True, alpha=0.2, linestyle="--")
    ax2.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"day15_online_comparison_{ticker}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 2: EWMA coefficient evolution ──────────────────────────
def plot_coef_evolution() -> None:
    fig, axes = plt.subplots(len(TICKERS), 1,
                              figsize=(13, 3.5 * len(TICKERS)))
    if len(TICKERS) == 1: axes = [axes]
    fig.suptitle("EWMA-Ridge: Coefficient Evolution Over Time",
                 fontsize=13, fontweight="bold")

    coef_colors = {
        "w_rv_lag_1"    : "#d62728",
        "w_rv_lag_5"    : "#ff7f0e",
        "w_rv_lag_21"   : "#1f77b4",
        "w_abs_ret_lag1": "#2ca02c",
    }

    for ax, ticker in zip(axes, TICKERS):
        path = os.path.join(OUT_DIR, f"day15_coef_history_{ticker}.csv")
        if not os.path.exists(path): continue
        df   = pd.read_csv(path, parse_dates=["date"])
        color= TICKER_COL[ticker]

        for col, c in coef_colors.items():
            if col in df.columns:
                ax.plot(df["step"], df[col],
                        color=c, linewidth=1.2,
                        label=col.replace("w_", ""))

        ax.axhline(0, color="black", linewidth=0.6,
                   linestyle=":", alpha=0.4)
        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.set_ylabel("Coefficient (standardised)")
        ax.legend(frameon=False, fontsize=8, ncol=2)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day15_coef_evolution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 3: PSI drift scores over time ──────────────────────────
def plot_drift_scores() -> None:
    n = len(TICKERS)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.5*n))
    if n == 1: axes = [axes]
    fig.suptitle("Population Stability Index (PSI) — RV Distribution Drift",
                 fontsize=13, fontweight="bold")

    for ax, ticker in zip(axes, TICKERS):
        path = os.path.join(OUT_DIR, f"day15_rv_drift_{ticker}.csv")
        if not os.path.exists(path): continue
        df   = pd.read_csv(path, parse_dates=["date"])
        color= TICKER_COL[ticker]

        ax.plot(df["date"], df["psi"],
                color=color, linewidth=1.3, alpha=0.9, label="PSI")
        ax.fill_between(df["date"], df["psi"],
                        color=color, alpha=0.15)

        # Threshold bands
        ax.axhline(0.10, color="orange", linewidth=1.0,
                   linestyle="--", alpha=0.7, label="Moderate (0.10)")
        ax.axhline(0.25, color="red",    linewidth=1.0,
                   linestyle="--", alpha=0.7, label="Major (0.25)")

        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.set_ylabel("PSI")
        ax.legend(frameon=False, fontsize=8, ncol=3)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines[["top","right"]].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day15_psi_drift_scores.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 4: Feature drift heatmap ───────────────────────────────
def plot_feature_drift_heatmap(ticker: str = "SPY") -> None:
    path = os.path.join(OUT_DIR, f"day15_feature_drift_{ticker}.csv")
    if not os.path.exists(path): return

    df   = pd.read_csv(path, parse_dates=["date"])
    psi_cols = [c for c in df.columns if c.startswith("psi_")]
    if not psi_cols: return

    matrix = df.set_index("date")[psi_cols].T
    matrix.index = [c.replace("psi_","") for c in matrix.index]

    fig, ax = plt.subplots(figsize=(12, max(3, len(psi_cols)*0.5)))
    sns.heatmap(
        matrix, ax=ax, cmap="YlOrRd",
        vmin=0, vmax=0.25,
        xticklabels=[d.strftime("%Y-%m") if i % 5 == 0 else ""
                     for i, d in enumerate(df["date"])],
        yticklabels=matrix.index,
        cbar_kws={"label": "PSI"},
        linewidths=0,
    )
    ax.set_title(f"{ticker} — Feature-Level PSI Over Time\n"
                 f"(yellow/red = significant drift from training distribution)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Date")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"day15_feature_drift_heatmap_{ticker}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 5: Monitoring dashboard ────────────────────────────────
def plot_monitoring_dashboard() -> None:
    path = os.path.join(OUT_DIR, "day15_monitoring_log.csv")
    if not os.path.exists(path): return

    df      = pd.read_csv(path, parse_dates=["date"])
    tickers = df["ticker"].unique()
    n       = len(tickers)

    fig, axes = plt.subplots(n, 1, figsize=(13, 4*n))
    if n == 1: axes = [axes]
    fig.suptitle("Model Monitoring Dashboard",
                 fontsize=13, fontweight="bold")

    for ax, ticker in zip(axes, tickers):
        sub   = df[df["ticker"]==ticker].sort_values("date")
        color = TICKER_COL.get(ticker, "steelblue")
        ax2   = ax.twinx()

        # Primary: rolling RMSE vs baseline
        ax.plot(sub["date"], sub["rmse"]*1e4,
                color=color, linewidth=1.3, label="Rolling RMSE")
        ax.plot(sub["date"], sub["baseline_rmse"]*1e4,
                color="black", linewidth=1.0, linestyle="--",
                alpha=0.5, label="Baseline")
        ax.plot(sub["date"], sub["alert_upper_rmse"]*1e4,
                color="red", linewidth=0.9, linestyle=":",
                alpha=0.7, label="Alert threshold")
        ax.set_ylabel("RMSE × 10⁴", color=color)

        # Secondary: PSI
        if "psi_rv" in sub.columns and sub["psi_rv"].notna().any():
            ax2.fill_between(sub["date"], sub["psi_rv"].fillna(0),
                             color="orange", alpha=0.2)
            ax2.plot(sub["date"], sub["psi_rv"].fillna(0),
                     color="orange", linewidth=0.9,
                     linestyle="-", alpha=0.7)
            ax2.set_ylabel("PSI", color="orange")
            ax2.tick_params(axis="y", labelcolor="orange")

        # Coloured background by alert level
        for _, row in sub.iterrows():
            c = ALERT_COL.get(row["alert_level"], "white")
            ax.axvspan(row["date"],
                       row["date"] + pd.Timedelta(days=20),
                       alpha=0.06, color=c)

        ax.set_title(f"{ticker}", fontsize=11, fontweight="bold")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines[["top"]].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        legend_patches = [
            mpatches.Patch(color=ALERT_COL["GREEN"],  alpha=0.3, label="GREEN"),
            mpatches.Patch(color=ALERT_COL["YELLOW"], alpha=0.3, label="YELLOW"),
            mpatches.Patch(color=ALERT_COL["RED"],    alpha=0.3, label="RED"),
        ]
        ax2.legend(handles=legend_patches, frameon=False, fontsize=7,
                   loc="upper right")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day15_monitoring_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 6: Error distribution comparison ───────────────────────
def plot_error_distributions(online_results: dict) -> None:
    n = len(TICKERS)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1: axes = [axes]
    fig.suptitle("Prediction Error Distribution: Online vs Offline",
                 fontsize=13, fontweight="bold")

    for ax, ticker in zip(axes, TICKERS):
        if ticker not in online_results: continue
        res  = online_results[ticker]
        y    = res["y"].values
        burn = 252

        for label, preds, color in [
            ("EWMA Online", res["preds_ewma"],    MODEL_COL["EWMA_Online"]),
            ("Offline 21d", res["preds_offline"],  MODEL_COL["Offline_21d"]),
        ]:
            valid = (~np.isnan(preds)) & (np.arange(len(preds)) >= burn)
            err   = (y - preds)[valid] * 100
            ax.hist(err, bins=50, density=True,
                    color=color, alpha=0.4, edgecolor="none", label=label)

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.set_xlabel("Forecast error (×100)")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day15_error_distributions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Main ────────────────────────────────────────────────────────
def run_all_plots(online_results: dict,
                   drift_results:  dict) -> None:
    print(f"\n{'='*55}")
    print("  DAY 15 — Generating Plots")
    print(f"{'='*55}")

    for ticker in TICKERS:
        plot_online_comparison(online_results, ticker=ticker)

    plot_coef_evolution()
    plot_drift_scores()

    for ticker in TICKERS:
        plot_feature_drift_heatmap(ticker=ticker)

    plot_monitoring_dashboard()
    plot_error_distributions(online_results)
    print("\n  All Day 15 plots complete.")