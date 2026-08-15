# ─────────────────────────────────────────────────────────────
# day18_plots.py
# All Day 18 visualisations:
#   1. Granger causality p-values (rolling and full-sample)
#   2. VAR coefficient matrix heatmap (spillover magnitudes)
#   3. Impulse Response Functions (9-panel, all pairs)
#   4. FEVD stacked bar chart (variance attribution per asset)
#   5. Rolling Total Connectedness Index with FROM/TO lines
#   6. Spillover forecast comparison bar chart
# ─────────────────────────────────────────────────────────────

import os
import warnings
warnings.filterwarnings("ignore")
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


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
PAIR_COL  = {
    "SPY_to_QQQ" : "#1f77b4", "QQQ_to_SPY" : "#aec7e8",
    "SPY_to_AAPL": "#ff7f0e", "AAPL_to_SPY": "#ffbb78",
    "QQQ_to_AAPL": "#2ca02c", "AAPL_to_QQQ": "#98df8a",
}


# ── Plot 1: Granger causality rolling p-values ──────────────────
def plot_rolling_granger() -> None:
    pairs = [("SPY","QQQ"), ("SPY","AAPL"), ("QQQ","AAPL")]
    n     = len(pairs)

    fig, axes = plt.subplots(n, 1, figsize=(13, 3.5*n))
    if n == 1: axes = [axes]
    fig.suptitle("Rolling Granger Causality p-values\n"
                 "(below 0.05 dashed line = significant at 5%)",
                 fontsize=13, fontweight="bold")

    for ax, (cause, effect) in zip(axes, pairs):
        # Both directions
        for direction in [(cause, effect), (effect, cause)]:
            c, e = direction
            key  = f"{c}_to_{e}"
            path = os.path.join(OUT_DIR, f"day18_rolling_granger_{key}.csv")
            if not os.path.exists(path): continue
            df    = pd.read_csv(path, parse_dates=["date"])
            color = PAIR_COL.get(key, "#888888")
            ax.plot(df["date"], df["p_value"],
                    color=color, linewidth=1.0, alpha=0.8,
                    label=f"{c}→{e}")

        ax.axhline(0.05, color="red", linewidth=1.0,
                   linestyle="--", alpha=0.7, label="p=0.05")
        ax.set_ylim(-0.02, 1.0)
        ax.set_ylabel("p-value")
        ax.set_title(f"{cause} ↔ {effect}", fontsize=11, fontweight="bold")
        ax.legend(frameon=False, fontsize=8, ncol=3)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines[["top","right"]].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day18_rolling_granger.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 2: VAR coefficient matrix heatmap ──────────────────────
def plot_var_coefficients() -> None:
    path = os.path.join(OUT_DIR, "day18_var_coefficients.csv")
    if not os.path.exists(path): return

    df = pd.read_csv(path, index_col=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        df.astype(float),
        ax=ax, annot=True, fmt=".4f",
        cmap="RdYlGn", center=0,
        linewidths=0.5,
        cbar_kws={"label": "Coefficient"},
    )
    ax.set_title("VAR(1) Coefficient Matrix\n"
                 "(row = receiving asset, col = sending asset lag-1)\n"
                 "Off-diagonal = spillover; Diagonal = own persistence",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Receiving asset (RV today)")
    ax.set_xlabel("Sending asset (RV yesterday)")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day18_var_coefficients.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 3: Impulse Response Functions ─────────────────────────
def plot_irf() -> None:
    path = os.path.join(OUT_DIR, "day18_irf.csv")
    if not os.path.exists(path): return

    df = pd.read_csv(path)

    fig, axes = plt.subplots(len(TICKERS), len(TICKERS),
                              figsize=(4.5*len(TICKERS), 3.5*len(TICKERS)),
                              sharex=True)
    fig.suptitle("Orthogonalised Impulse Response Functions\n"
                 "(row = responding asset, col = shocked asset)",
                 fontsize=13, fontweight="bold")

    for i, resp in enumerate(TICKERS):
        for j, shock in enumerate(TICKERS):
            ax  = axes[i][j]
            sub = df[(df["response"]==resp) & (df["shock"]==shock)]
            if sub.empty:
                continue
            color = TC[shock]
            ax.plot(sub["period"], sub["irf"],
                    color=color, linewidth=1.8)
            ax.fill_between(sub["period"], sub["irf"], 0,
                            alpha=0.15, color=color)
            ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
            if i == 0:
                ax.set_title(f"Shock: {shock}", fontsize=10,
                             fontweight="bold", color=color)
            if j == 0:
                ax.set_ylabel(f"Response:\n{resp}", fontsize=9)
            ax.grid(True, alpha=0.15, linestyle="--")
            ax.spines[["top","right"]].set_visible(False)
            if i == len(TICKERS)-1:
                ax.set_xlabel("Days after shock")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day18_irf.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 4: FEVD stacked bar chart ──────────────────────────────
def plot_fevd() -> None:
    path = os.path.join(OUT_DIR, "day18_fevd.csv")
    if not os.path.exists(path): return

    df = pd.read_csv(path, index_col=0)
    # Keep only the n×n asset block (exclude TO/FROM/NET/TCI rows)
    asset_rows = [r for r in df.index if any(t in r for t in TICKERS)]
    asset_cols = [c for c in df.columns if any(t in c for t in TICKERS)]
    mat = df.loc[asset_rows[:len(TICKERS)], asset_cols[:len(TICKERS)]]
    mat.index   = TICKERS
    mat.columns = TICKERS

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = [TC[t] for t in TICKERS]
    bottom  = np.zeros(len(TICKERS))
    for j, (source, color) in enumerate(zip(TICKERS, colors)):
        vals = mat.iloc[:, j].values.astype(float)
        ax.bar(TICKERS, vals, bottom=bottom, label=source,
               color=color, alpha=0.82, edgecolor="white")
        bottom += vals

    ax.set_ylabel("Fraction of forecast error variance")
    ax.set_title(f"Forecast Error Variance Decomposition\n"
                 f"at {10}-step horizon (colour = source of shock)",
                 fontsize=12, fontweight="bold")
    ax.legend(title="Shock from", frameon=False, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2, linestyle="--", axis="y")
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day18_fevd.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 5: Rolling TCI ─────────────────────────────────────────
def plot_rolling_tci() -> None:
    path = os.path.join(OUT_DIR, "day18_rolling_tci.csv")
    if not os.path.exists(path): return

    df = pd.read_csv(path, index_col=0, parse_dates=True)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle("Rolling Diebold-Yilmaz Connectedness (252-day window)",
                 fontsize=13, fontweight="bold")

    # Top panel: TCI
    ax1 = axes[0]
    ax1.fill_between(df.index, df["TCI"], alpha=0.3, color="steelblue")
    ax1.plot(df.index, df["TCI"], color="steelblue", linewidth=1.3)
    ax1.axhline(df["TCI"].mean(), color="red", linewidth=1.0,
                linestyle="--", alpha=0.7, label=f"Mean={df['TCI'].mean():.3f}")
    ax1.set_ylabel("Total Connectedness Index")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(True, alpha=0.2, linestyle="--")
    ax1.spines[["top","right"]].set_visible(False)

    # Bottom panel: FROM scores per asset
    ax2 = axes[1]
    for ticker in TICKERS:
        col = f"FROM_{ticker}"
        if col in df.columns:
            ax2.plot(df.index, df[col], color=TC[ticker],
                     linewidth=1.3, alpha=0.85, label=ticker)

    ax2.set_ylabel("FROM score (vol received from others)")
    ax2.set_xlabel("Date")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(True, alpha=0.2, linestyle="--")
    ax2.spines[["top","right"]].set_visible(False)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day18_rolling_tci.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 6: Spillover forecast comparison ───────────────────────
def plot_spillover_comparison() -> None:
    path = os.path.join(OUT_DIR, "day18_spillover_metrics.csv")
    if not os.path.exists(path): return

    df      = pd.read_csv(path)
    models  = ["HAR", "HAR+Spillover", "HAR+TCI"]
    tickers = sorted(df["Ticker"].unique())
    x       = np.arange(len(tickers))
    w       = 0.25
    colors  = ["#7f7f7f", "#1f77b4", "#9467bd"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Spillover-Augmented Forecasting vs HAR Baseline",
                 fontsize=13, fontweight="bold")

    for ax, metric, scale, label in [
        (axes[0], "RMSE",  1e4, "RMSE × 10⁴"),
        (axes[1], "QLIKE", 1.0, "QLIKE"),
    ]:
        for i, (model, color) in enumerate(zip(models, colors)):
            vals = [df[(df["Ticker"]==t) & (df["Model"]==model)][metric].values[0]
                    if len(df[(df["Ticker"]==t) & (df["Model"]==model)]) > 0
                    else np.nan
                    for t in tickers]
            ax.bar(x + i*w, np.array(vals)*scale, width=w,
                   label=model, color=color, alpha=0.85, edgecolor="white")

        ax.set_xticks(x + w)
        ax.set_xticklabels(tickers, fontsize=11)
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.2, linestyle="--", axis="y")
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day18_spillover_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Main ────────────────────────────────────────────────────────
def run_all_plots() -> None:
    print(f"\n{'='*55}")
    print("  DAY 18 — Generating Plots")
    print(f"{'='*55}")

    plot_rolling_granger()
    plot_var_coefficients()
    plot_irf()
    plot_fevd()
    plot_rolling_tci()
    plot_spillover_comparison()

    print("\n  All Day 18 plots complete.")