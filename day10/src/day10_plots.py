# ─────────────────────────────────────────────────────────────
# day10_plots.py
# All Day 10 visualisations:
#   1. Forecast combination RMSE comparison (individual + 3 combos)
#   2. Combination weights bar chart (InvRMSE vs Granger-Ramanathan)
#   3. VaR backtest: actual returns + VaR line + violation markers
#   4. Kupiec/Christoffersen test results heatmap
#   5. SHAP summary plot (beeswarm-style, manual matplotlib)
#   6. SHAP dependence plot (top feature vs SHAP value)
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

MODEL_COLORS = {
    "HAR": "#1f77b4", "GARCH": "#ff7f0e",
    "XGBoost": "#2ca02c", "HARNet": "#9467bd",
    "Simple": "#7f7f7f", "InvRMSE": "#e377c2",
    "GrangerRam": "#8c564b",
}
TICKER_COL = {"SPY": "#1f77b4", "QQQ": "#ff7f0e", "AAPL": "#2ca02c"}
TICKERS    = ["SPY", "QQQ", "AAPL"]


# ── Plot 1: Combination RMSE comparison ─────────────────────────
def plot_combination_comparison(summary_csv: str) -> None:
    df = pd.read_csv(summary_csv)
    rmse_cols = [c for c in df.columns if c.startswith("RMSE_")]
    labels    = [c.replace("RMSE_", "") for c in rmse_cols]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df))
    w = 0.8 / len(rmse_cols)

    for i, (col, label) in enumerate(zip(rmse_cols, labels)):
        color = MODEL_COLORS.get(label, "#888888")
        ax.bar(x + i * w, df[col] * 1e4, width=w,
               label=label, color=color, alpha=0.85,
               edgecolor="white")

    ax.set_xticks(x + w * (len(rmse_cols) - 1) / 2)
    ax.set_xticklabels(df["Ticker"], fontsize=11)
    ax.set_ylabel("RMSE × 10⁴")
    ax.set_title("Forecast Combination: Individual Models vs Ensembles",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    ax.grid(True, alpha=0.25, linestyle="--", axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day10_combination_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 2: Combination weights ─────────────────────────────────
def plot_combination_weights(weight_csv: str) -> None:
    df = pd.read_csv(weight_csv)
    methods = df["Method"].unique()

    fig, axes = plt.subplots(1, len(methods), figsize=(6*len(methods), 5))
    if len(methods) == 1:
        axes = [axes]
    fig.suptitle("Forecast Combination Weights by Method",
                 fontsize=13, fontweight="bold")

    for ax, method in zip(axes, methods):
        sub     = df[df["Method"] == method]
        pivot   = sub.pivot(index="Model", columns="Ticker", values="Weight")
        pivot   = pivot.reindex(["HAR", "GARCH", "XGBoost", "HARNet"])

        pivot.plot(kind="bar", ax=ax,
                   color=[TICKER_COL[t] for t in pivot.columns],
                   alpha=0.85, edgecolor="white")
        ax.set_title(method, fontsize=11, fontweight="bold")
        ax.set_ylabel("Weight")
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.legend(title="Ticker", frameon=False, fontsize=8)
        ax.grid(True, alpha=0.25, linestyle="--", axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0.25, color="black", linewidth=0.7,
                   linestyle=":", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day10_combination_weights.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 3: VaR backtest time series ────────────────────────────
def plot_var_backtest(var_series: dict, alpha: float = 0.05) -> None:
    n = len(TICKERS)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.5 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"VaR Backtest at {int((1-alpha)*100)}% Confidence",
                 fontsize=13, fontweight="bold")

    for ax, ticker in zip(axes, TICKERS):
        if ticker not in var_series or alpha not in var_series[ticker]:
            continue
        s = var_series[ticker][alpha]
        color = TICKER_COL[ticker]

        ax.plot(s["dates"], s["actual"] * 100,
                color=color, linewidth=0.8, alpha=0.7,
                label="Actual return")
        ax.plot(s["dates"], s["VaR"] * 100,
                color="red", linewidth=1.2,
                linestyle="--", label="VaR threshold")

        viol_mask = s["violations"].astype(bool)
        ax.scatter(
            np.array(s["dates"])[viol_mask],
            s["actual"][viol_mask] * 100,
            color="darkred", s=25, zorder=5,
            label=f"Violations (n={viol_mask.sum()})"
        )

        ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.set_ylabel("Daily Return (%)")
        ax.legend(frameon=False, fontsize=8, ncol=3)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    out = os.path.join(
        OUT_DIR, f"day10_var_backtest_{int(alpha*100)}.png"
    )
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 4: Kupiec/Christoffersen heatmap ───────────────────────
def plot_var_test_heatmap(var_csv: str) -> None:
    df = pd.read_csv(var_csv)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("VaR Backtest Statistical Tests",
                 fontsize=13, fontweight="bold")

    for ax, col, title in zip(
        axes,
        ["Kupiec_p", "Christoffersen_p"],
        ["Kupiec POF Test (calibration)",
         "Christoffersen Test (independence)"]
    ):
        pivot = df.pivot_table(
            index="Confidence", columns="Ticker",
            values=col, aggfunc="mean"
        )
        sns.heatmap(
            pivot, ax=ax, annot=True, fmt=".3f",
            cmap="RdYlGn", vmin=0, vmax=1,
            linewidths=0.5,
            cbar_kws={"label": "p-value"},
        )
        ax.set_title(title, fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day10_var_test_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 5: SHAP summary (manual beeswarm) ──────────────────────
def plot_shap_summary(shap_results: dict, ticker: str = "SPY") -> None:
    if ticker not in shap_results or not shap_results[ticker].get("shap_available"):
        print(f"  ⚠ No SHAP data for {ticker}")
        return

    res    = shap_results[ticker]
    sv     = res["shap_values"]
    X_test = res["X_test"]

    top_feats = res["importance_df"].head(10)["feature"].tolist()
    feat_idx  = [X_test.columns.get_loc(f) for f in top_feats]

    fig, ax = plt.subplots(figsize=(9, 6))

    for i, (feat, idx) in enumerate(zip(top_feats, feat_idx)):
        shap_vals = sv.values[:, idx]
        feat_vals = X_test.iloc[:, idx].values

        # Normalise feature values to [0,1] for colour mapping
        fmin, fmax = feat_vals.min(), feat_vals.max()
        norm_vals = (feat_vals - fmin) / (fmax - fmin + 1e-10)

        y_jitter = i + np.random.uniform(-0.3, 0.3, len(shap_vals))
        sc = ax.scatter(
            shap_vals, y_jitter,
            c=norm_vals, cmap="coolwarm",
            s=10, alpha=0.6, edgecolors="none"
        )

    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title(f"{ticker} — SHAP Summary (top 10 features)\n"
                 f"blue=low feature value, red=high feature value",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, linestyle="--", axis="x")
    ax.spines[["top", "right"]].set_visible(False)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Feature value (normalised)")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"day10_shap_summary_{ticker}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 6: SHAP dependence plot ────────────────────────────────
def plot_shap_dependence(shap_results: dict,
                          ticker: str = "SPY",
                          feature: str = None) -> None:
    if ticker not in shap_results or not shap_results[ticker].get("shap_available"):
        return

    res    = shap_results[ticker]
    sv     = res["shap_values"]
    X_test = res["X_test"]

    if feature is None:
        feature = res["importance_df"].iloc[0]["feature"]

    if feature not in X_test.columns:
        return

    idx       = X_test.columns.get_loc(feature)
    feat_vals = X_test[feature].values
    shap_vals = sv.values[:, idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(feat_vals, shap_vals,
               color="#1f77b4", alpha=0.4, s=15, edgecolors="none")

    # Add a smoothed trend line (simple rolling mean over sorted x)
    order = np.argsort(feat_vals)
    x_sorted = feat_vals[order]
    y_sorted = shap_vals[order]
    window   = max(10, len(x_sorted) // 20)
    y_smooth = pd.Series(y_sorted).rolling(
        window, min_periods=1, center=True
    ).mean()
    ax.plot(x_sorted, y_smooth, color="red", linewidth=2,
            label="Smoothed trend")

    ax.axhline(0, color="black", linewidth=0.7, alpha=0.4)
    ax.set_xlabel(feature)
    ax.set_ylabel("SHAP value")
    ax.set_title(f"{ticker} — SHAP Dependence: {feature}",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(
        OUT_DIR, f"day10_shap_dependence_{ticker}_{feature}.png"
    )
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Main ────────────────────────────────────────────────────────
def run_all_plots(var_series: dict = None,
                   shap_results: dict = None):
    print(f"\n{'='*55}")
    print("  DAY 10 — Generating Plots")
    print(f"{'='*55}")

    combo_summary = os.path.join(OUT_DIR, "day10_combination_summary.csv")
    combo_weights = os.path.join(OUT_DIR, "day10_combination_weights.csv")
    var_csv       = os.path.join(OUT_DIR, "day10_var_backtest.csv")

    if os.path.exists(combo_summary):
        plot_combination_comparison(combo_summary)
    if os.path.exists(combo_weights):
        plot_combination_weights(combo_weights)
    if os.path.exists(var_csv):
        plot_var_test_heatmap(var_csv)

    if var_series:
        plot_var_backtest(var_series, alpha=0.05)
        plot_var_backtest(var_series, alpha=0.01)

    if shap_results:
        for ticker in TICKERS:
            if ticker in shap_results:
                plot_shap_summary(shap_results, ticker=ticker)
                plot_shap_dependence(shap_results, ticker=ticker)

    print("\n  All Day 10 plots complete.")