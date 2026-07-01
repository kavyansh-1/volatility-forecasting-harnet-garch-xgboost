# ─────────────────────────────────────────────────────────────
# day13_plots.py
# All Day 13 visualisations:
#   1. Training curves for all 3 architectures × 3 tickers
#   2. Forecast vs actual (all three architectures, one ticker)
#   3. Attention weight heatmap (22×22 for one test prediction)
#   4. Average attention by lag position (bar chart)
#   5. RMSE comparison: Day 13 models vs Day 5 HARNet
#   6. Attention entropy over time (does attention concentrate?)
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

TICKERS   = ["SPY", "QQQ", "AAPL"]
ARCH_COL  = {
    "TemporalAttention"   : "#1f77b4",
    "TemporalAttentionPos": "#ff7f0e",
    "Transformer"         : "#9467bd",
    "HARNet_Day5"         : "#2ca02c",
}
TICKER_COL = {"SPY": "#1f77b4", "QQQ": "#ff7f0e", "AAPL": "#2ca02c"}


# ── Plot 1: Training curves ──────────────────────────────────────
def plot_training_curves(history_csv: str) -> None:
    df      = pd.read_csv(history_csv)
    archs   = df["arch"].unique()
    tickers = df["ticker"].unique()

    fig, axes = plt.subplots(len(tickers), len(archs),
                              figsize=(5*len(archs), 4*len(tickers)),
                              sharey=False)
    fig.suptitle("Day 13 — Training & Validation Loss",
                 fontsize=13, fontweight="bold")

    for i, ticker in enumerate(tickers):
        for j, arch in enumerate(archs):
            ax  = axes[i][j]
            sub = df[(df["ticker"]==ticker) & (df["arch"]==arch)]
            col = ARCH_COL.get(arch, "#888888")
            ax.plot(sub["epoch"], sub["train_loss"],
                    color=col, linewidth=1.5, label="Train")
            ax.plot(sub["epoch"], sub["val_loss"],
                    color=col, linewidth=1.5, linestyle="--",
                    alpha=0.7, label="Val")
            ax.set_title(f"{ticker} – {arch}", fontsize=8,
                         fontweight="bold")
            if j == 0:
                ax.set_ylabel("MSE Loss", fontsize=8)
            ax.legend(frameon=False, fontsize=7)
            ax.grid(True, alpha=0.2, linestyle="--")
            ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day13_training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 2: Forecast vs actual ───────────────────────────────────
def plot_forecast_vs_actual(pred_csv: str,
                             ticker:   str = "SPY") -> None:
    df  = pd.read_csv(pred_csv)
    sub = df[df["ticker"] == ticker]

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(range(len(sub[sub["arch"]==sub["arch"].iloc[0]])),
            sub[sub["arch"]==sub["arch"].iloc[0]]["actual"] * 100,
            color="red", linewidth=0.8, alpha=0.7, label="Actual RV")

    for arch in sub["arch"].unique():
        a_sub = sub[sub["arch"]==arch].reset_index(drop=True)
        ax.plot(range(len(a_sub)), a_sub["pred"] * 100,
                color=ARCH_COL.get(arch, "#888888"),
                linewidth=1.0, alpha=0.85, label=arch)

    ax.set_title(f"{ticker} — All Architectures: Forecast vs Actual RV",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Annualised RV (%)")
    ax.set_xlabel("Test observation index")
    ax.legend(frameon=False, fontsize=8, ncol=4)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"day13_forecast_vs_actual_{ticker}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 3: Attention heatmap ─────────────────────────────────────
def plot_attention_heatmap(all_results: dict,
                            ticker: str = "SPY",
                            arch:   str = "Transformer") -> None:
    """
    Visualise the 22×22 attention matrix for one test prediction.
    Rows = query positions (which timestep is querying)
    Cols = key positions (which timestep is being attended to)
    """
    if ticker not in all_results or arch not in all_results[ticker]:
        return

    attn = all_results[ticker][arch].get("attn")
    if attn is None or len(attn) == 0:
        return

    # Use first available batch's attention matrix
    attn_matrix = attn[0] if attn.ndim == 3 else attn

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        attn_matrix, ax=ax,
        cmap="Blues", vmin=0, vmax=attn_matrix.max(),
        xticklabels=[f"t-{22-i}" for i in range(22)],
        yticklabels=[f"t-{22-i}" for i in range(22)],
        linewidths=0, cbar_kws={"label": "Attention weight"},
    )
    ax.set_title(f"{ticker} {arch} — Attention Map\n"
                 f"(Row i = how step i attends to all other steps)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Key position (timestep being attended to)")
    ax.set_ylabel("Query position (current timestep)")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)

    plt.tight_layout()
    out = os.path.join(OUT_DIR,
                        f"day13_attention_heatmap_{ticker}_{arch}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 4: Average attention by lag ─────────────────────────────
def plot_avg_attention_by_lag(attn_df: pd.DataFrame) -> None:
    if attn_df.empty:
        return

    tickers = attn_df["Ticker"].unique()
    archs   = attn_df["Arch"].unique()

    fig, axes = plt.subplots(len(tickers), 1,
                              figsize=(10, 4*len(tickers)))
    if len(tickers) == 1:
        axes = [axes]
    fig.suptitle("Average Attention Weight by Lag Position\n"
                 "(Which past days does each model focus on?)",
                 fontsize=13, fontweight="bold")

    for ax, ticker in zip(axes, tickers):
        sub = attn_df[attn_df["Ticker"] == ticker]
        for arch in archs:
            a_sub = sub[sub["Arch"] == arch].sort_values("Lag")
            if a_sub.empty:
                continue
            ax.plot(a_sub["Lag"], a_sub["Avg_Attn"],
                    color=ARCH_COL.get(arch, "#888888"),
                    linewidth=1.5, marker="o", markersize=3,
                    label=arch)

        # Mark HAR reference lags
        for lag, label in [(1, "Daily"), (5, "Weekly"), (22, "Monthly")]:
            ax.axvline(lag, color="red", linewidth=0.8,
                       linestyle=":", alpha=0.5)
            ax.text(lag + 0.3, ax.get_ylim()[1] * 0.9,
                    label, fontsize=7, color="red", alpha=0.7)

        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.set_xlabel("Lag (days back)")
        ax.set_ylabel("Avg attention received")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day13_avg_attention_by_lag.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 5: RMSE comparison chart ───────────────────────────────
def plot_rmse_comparison(metrics_csv: str) -> None:
    df = pd.read_csv(metrics_csv)

    tickers = sorted(df["Ticker"].unique())
    models  = df["Model"].unique()
    x       = np.arange(len(tickers))
    w       = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, model in enumerate(models):
        vals = [
            df[(df["Ticker"]==t) & (df["Model"]==model)]["RMSE"].values[0]
            if len(df[(df["Ticker"]==t) & (df["Model"]==model)]) > 0
            else np.nan
            for t in tickers
        ]
        color = ARCH_COL.get(model, "#888888")
        ax.bar(x + i * w, np.array(vals) * 1e4, width=w,
               label=model, color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x + w * (len(models)-1)/2)
    ax.set_xticklabels(tickers, fontsize=11)
    ax.set_ylabel("RMSE × 10⁴")
    ax.set_title("Day 13 Attention Models vs Day 5 HARNet Baseline",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=4)
    ax.grid(True, alpha=0.25, linestyle="--", axis="y")
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day13_rmse_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 6: Attention entropy over time ─────────────────────────
def plot_attention_entropy(all_results: dict,
                            ticker: str = "SPY") -> None:
    """
    Entropy of the attention distribution at each test step.
    Low entropy = attention concentrated on few timesteps (focused).
    High entropy = attention spread evenly (uncertain about which lag matters).
    Entropy = -sum(attn * log(attn + eps))
    """
    if ticker not in all_results:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title(f"{ticker} — Attention Entropy Over Test Period\n"
                 "(low = focused on few lags, high = diffuse)",
                 fontsize=12, fontweight="bold")

    plotted = False
    for arch, res in all_results[ticker].items():
        attn = res.get("attn")
        if attn is None or attn.ndim < 2:
            continue

        # attn shape: (n_test_batches, T, T)
        # Entropy of each row (query position) averaged per step
        eps      = 1e-8
        if attn.ndim == 3:
            entropy  = -(attn * np.log(attn + eps)).sum(axis=-1)
            avg_ent  = entropy.mean(axis=-1)   # avg over query positions
        else:
            entropy  = -(attn * np.log(attn + eps)).sum(axis=-1)
            avg_ent  = entropy

        ax.plot(range(len(avg_ent)), avg_ent,
                color=ARCH_COL.get(arch, "#888888"),
                linewidth=1.0, alpha=0.8, label=arch)
        plotted = True

    if plotted:
        ax.set_xlabel("Test observation index")
        ax.set_ylabel("Attention entropy")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR,
                        f"day13_attention_entropy_{ticker}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Main ────────────────────────────────────────────────────────
def run_all_plots(all_results: dict, attn_df: pd.DataFrame) -> None:
    print(f"\n{'='*55}")
    print("  DAY 13 — Generating Plots")
    print(f"{'='*55}")

    history_csv = os.path.join(OUT_DIR, "day13_training_history.csv")
    pred_csv    = os.path.join(OUT_DIR, "day13_predictions.csv")
    metrics_csv = os.path.join(OUT_DIR, "day13_all_metrics.csv")

    if os.path.exists(history_csv):
        plot_training_curves(history_csv)

    for ticker in TICKERS:
        if os.path.exists(pred_csv):
            plot_forecast_vs_actual(pred_csv, ticker=ticker)
        plot_attention_entropy(all_results, ticker=ticker)

    for ticker in TICKERS:
        for arch in ["Transformer", "TemporalAttentionPos"]:
            plot_attention_heatmap(all_results, ticker=ticker, arch=arch)

    if not attn_df.empty:
        plot_avg_attention_by_lag(attn_df)

    if os.path.exists(metrics_csv):
        plot_rmse_comparison(metrics_csv)

    print("\n  All Day 13 plots complete.")