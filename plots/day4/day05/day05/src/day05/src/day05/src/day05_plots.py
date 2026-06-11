# ─────────────────────────────────────────────────────────────
# day05_plots.py
# Generates all Day 5 plots:
#   1. Training + validation loss curves
#   2. HARNet forecast vs actual RV
#   3. Four-model RMSE + DirAcc bar chart
#   4. Residual error distribution
#   5. QLIKE comparison bar chart
# ─────────────────────────────────────────────────────────────

import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def _find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "data" / "processed").is_dir() and (candidate / "reports").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root")


REPO_ROOT = _find_repo_root()
OUT_DIR = REPO_ROOT / "plots" / "day4" / "day05" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "HAR"    : "#1f77b4",
    "GARCH"  : "#ff7f0e",
    "XGBoost": "#2ca02c",
    "HARNet" : "#9467bd",
    "Actual" : "#d62728",
}
TICKER_COL = {"SPY": "#1f77b4", "QQQ": "#ff7f0e", "AAPL": "#2ca02c"}


# ── Plot 1: Training curves ─────────────────────────────────────
def plot_training_curves(history_csv: str) -> None:
    df      = pd.read_csv(history_csv)
    tickers = df["ticker"].unique()
    fig, axes = plt.subplots(1, len(tickers),
                              figsize=(5 * len(tickers), 4))
    if len(tickers) == 1:
        axes = [axes]
    fig.suptitle("HARNet — Training & Validation Loss",
                 fontsize=13, fontweight="bold")

    for ax, ticker in zip(axes, tickers):
        sub   = df[df["ticker"] == ticker]
        color = TICKER_COL.get(ticker, "steelblue")
        ax.plot(sub["epoch"], sub["train_loss"],
                color=color, linewidth=1.5, label="Train")
        ax.plot(sub["epoch"], sub["val_loss"],
                color=color, linewidth=1.5,
                linestyle="--", alpha=0.75, label="Val")
        ax.set_title(ticker, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "day05_training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 2: Forecast vs actual ──────────────────────────────────
def plot_forecast_vs_actual(pred_csv: str) -> None:
    df      = pd.read_csv(pred_csv)
    tickers = df["ticker"].unique()
    fig, axes = plt.subplots(len(tickers), 1,
                              figsize=(13, 4 * len(tickers)))
    if len(tickers) == 1:
        axes = [axes]
    fig.suptitle("HARNet — Forecast vs Realised RV (Test Set)",
                 fontsize=13, fontweight="bold")

    for ax, ticker in zip(axes, tickers):
        sub = df[df["ticker"] == ticker].reset_index(drop=True)
        x   = np.arange(len(sub))
        ax.plot(x, sub["actual"]      * 100,
                color=PALETTE["Actual"],  linewidth=0.9,
                alpha=0.8, label="Realised RV")
        ax.plot(x, sub["harnet_pred"] * 100,
                color=PALETTE["HARNet"], linewidth=1.1,
                alpha=0.85, label="HARNet")
        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.set_ylabel("Annualised RV (%)")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Test observation index")
    plt.tight_layout()
    out = OUT_DIR / "day05_forecast_vs_actual.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 3: Four-model comparison ───────────────────────────────
def plot_model_comparison(metrics_csv: str) -> None:
    df      = pd.read_csv(metrics_csv)
    tickers = sorted(df["Ticker"].unique())
    models  = df["Model"].unique()
    x       = np.arange(len(tickers))
    w       = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("All Models — Out-of-Sample Comparison",
                 fontsize=13, fontweight="bold")

    for ax, (metric, scale, label) in zip(axes, [
        ("RMSE",   1e4, "RMSE  (×10⁴)"),
        ("DirAcc",  1,  "Directional Accuracy (%)"),
    ]):
        for i, model in enumerate(models):
            vals = []
            for t in tickers:
                sub = df[(df["Ticker"] == t) & (df["Model"] == model)]
                vals.append(float(sub[metric].iloc[0])
                            if len(sub) else np.nan)
            color = PALETTE.get(model, "#888888")
            ax.bar(x + i * w, np.array(vals) * scale,
                   width=w, label=model, color=color,
                   alpha=0.85, edgecolor="white")

        ax.set_xticks(x + w * (len(models) - 1) / 2)
        ax.set_xticklabels(tickers, fontsize=11)
        ax.set_ylabel(label)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.25, linestyle="--", axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        if metric == "DirAcc":
            ax.axhline(50, color="red", linestyle="--",
                       linewidth=0.9, alpha=0.5)

    plt.tight_layout()
    out = OUT_DIR / "day05_model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 4: Residuals ───────────────────────────────────────────
def plot_residuals(pred_csv: str) -> None:
    df      = pd.read_csv(pred_csv)
    tickers = df["ticker"].unique()
    fig, axes = plt.subplots(1, len(tickers),
                              figsize=(5 * len(tickers), 4))
    if len(tickers) == 1:
        axes = [axes]
    fig.suptitle("HARNet — Forecast Error Distribution",
                 fontsize=13, fontweight="bold")

    for ax, ticker in zip(axes, tickers):
        sub   = df[df["ticker"] == ticker]
        resid = (sub["actual"] - sub["harnet_pred"]).values
        color = TICKER_COL.get(ticker, "steelblue")

        ax.hist(resid * 100, bins=60, density=True,
                color=color, alpha=0.55, edgecolor="none")
        xr  = np.linspace(resid.min(), resid.max(), 300) * 100
        pdf = stats.norm.pdf(
            xr,
            loc=resid.mean() * 100,
            scale=max(resid.std() * 100, 1e-8),
        )
        ax.plot(xr, pdf, color="black", linewidth=1.6,
                linestyle="--", label="Normal fit")
        ax.set_title(f"{ticker}  σ={resid.std()*100:.4f}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Error (×100)")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "day05_residuals.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 5: QLIKE comparison ────────────────────────────────────
def plot_qlike_comparison(metrics_csv: str) -> None:
    df      = pd.read_csv(metrics_csv)
    # Filter extreme GARCH QLIKE values for readable scale
    df_plot = df[df["QLIKE"].abs() < 100].copy()
    if df_plot.empty:
        df_plot = df.copy()
    tickers = sorted(df_plot["Ticker"].unique())
    models  = df_plot["Model"].unique()
    x       = np.arange(len(tickers))
    w       = 0.2

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, model in enumerate(models):
        vals = []
        for t in tickers:
            sub = df_plot[(df_plot["Ticker"] == t) &
                          (df_plot["Model"]  == model)]
            vals.append(float(sub["QLIKE"].iloc[0])
                        if len(sub) else np.nan)
        color = PALETTE.get(model, "#888888")
        ax.bar(x + i * w, vals, width=w, label=model,
               color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels(tickers, fontsize=11)
    ax.set_ylabel("QLIKE  (lower = better)")
    ax.set_title("QLIKE Loss — All Models",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--", axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = OUT_DIR / "day05_qlike_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Main ────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*55}")
    print("  DAY 5 — Generating Plots")
    print(f"{'='*55}")

    history_csv = OUT_DIR / "day05_training_history.csv"
    pred_csv    = OUT_DIR / "day05_harnet_predictions.csv"
    metrics_csv = OUT_DIR / "day05_all_model_metrics.csv"

    if os.path.exists(history_csv):
        plot_training_curves(history_csv)
    if os.path.exists(pred_csv):
        plot_forecast_vs_actual(pred_csv)
        plot_residuals(pred_csv)
    if os.path.exists(metrics_csv):
        plot_model_comparison(metrics_csv)
        plot_qlike_comparison(metrics_csv)

    print("\n  All Day 5 plots complete.")


if __name__ == "__main__":
    main()