"""
day12/src/plot_garch_volatility.py
-----------------------------------
Days 12-13:
  Plot 1 — Returns time series + GARCH conditional volatility overlaid
  Plot 2 — Realised volatility vs GARCH forecast volatility overlaid

Imports directly from your existing modules:
  - models/tune_garch_models.py  → tune_garch_all_tickers()
  - data/processed/              → your Day 2 processed CSVs

Run from project root:
    python day12/src/plot_garch_volatility.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

# ── path setup — same pattern as your other day files ────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".vscode"))

from models.feature_engineering import load_ticker_frames
from models.tune_garch_models   import tune_garch_all_tickers

# ── constants ─────────────────────────────────────────────────────────────────
TICKERS = ["SPY", "QQQ", "AAPL"]

# matches your existing project colour palette
PALETTE = {
    "SPY" : "#1f77b4",
    "QQQ" : "#ff7f0e",
    "AAPL": "#2ca02c",
}

STYLE = "seaborn-v0_8-whitegrid"
DPI   = 150

# output folder — same pattern as day08/output, day09/output etc.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)


# ── helper: build vol series from arch result object ─────────────────────────
def extract_vol_series(best_result, returns: pd.Series) -> pd.DataFrame:
    """
    From your tune_garch best_result object, extract:
      - conditional_vol  : GARCH fitted conditional volatility (annualised %)
      - realised_vol     : 20-day rolling realised vol (annualised %)
      - log_return       : raw log returns

    best_result is the arch ARCHModelResult object stored in
    tune_garch() under key '_result' (passed back as 'best_result').

    Returns a clean DataFrame aligned on the return index.
    """
    # conditional volatility from arch result — daily, in % (because fit_garch
    # scales returns by *100 before fitting, so vol is already in % units)
    cond_vol_pct = best_result.conditional_volatility          # daily %
    cond_vol_ann = cond_vol_pct * np.sqrt(252)                 # annualised %

    # align to returns index — arch trims first few rows during fitting
    r = returns.dropna()
    cond_vol_ann = cond_vol_ann.reindex(r.index)

    # realised vol: 20-day rolling std of returns, annualised, in %
    realised_ann = r.rolling(20, min_periods=20).std() * np.sqrt(252) * 100

    df = pd.DataFrame({
        "log_return"     : r * 100,          # in %
        "conditional_vol": cond_vol_ann,
        "realised_vol"   : realised_ann,
    }, index=r.index).dropna(subset=["conditional_vol"])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Returns + GARCH Conditional Volatility
# ─────────────────────────────────────────────────────────────────────────────
def plot_returns_and_garch_vol(
    df         : pd.DataFrame,
    ticker     : str,
    best_row   : pd.Series,
) -> None:
    """
    Two-panel chart per ticker:
      Top    → daily log returns as bar chart (green = positive, red = negative)
      Bottom → GARCH conditional volatility as line + shaded area

    What this shows:
      The classic volatility clustering picture — during turbulent periods
      both the return bars AND the vol line are large simultaneously.
      The vol line rises during return spikes and decays slowly after,
      which is exactly what the beta (persistence) parameter controls.
    """
    color       = PALETTE[ticker]
    vol_model   = best_row["vol_model"]
    p           = int(best_row["p"])
    q           = int(best_row["q"])
    dist        = best_row["dist"]
    persistence = round(best_row["persistence"], 4)

    with plt.style.context(STYLE):
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle(
            f"{ticker}  —  Daily Returns and "
            f"{vol_model}({p},{q})-{dist.upper()} Conditional Volatility\n"
            f"Persistence (α+β) = {persistence}",
            fontsize=13, fontweight="bold", y=1.01,
        )

        gs     = gridspec.GridSpec(2, 1, height_ratios=[1.4, 1], hspace=0.06)
        ax_ret = fig.add_subplot(gs[0])
        ax_vol = fig.add_subplot(gs[1], sharex=ax_ret)

        # ── top panel: returns ────────────────────────────────────────────────
        ret         = df["log_return"]
        bar_colors  = [color if v >= 0 else "#d62728" for v in ret]
        ax_ret.bar(ret.index, ret, color=bar_colors,
                   alpha=0.65, width=1.0, linewidth=0)
        ax_ret.axhline(0, color="black", linewidth=0.7, zorder=5)
        ax_ret.set_ylabel("Log Return (%)", fontsize=11)
        ax_ret.spines[["top", "right"]].set_visible(False)
        ax_ret.tick_params(labelbottom=False, labelsize=9)

        ax_ret.text(
            0.01, 0.97,
            f"{color[1:]}  = positive return     red = negative return\n"
            "Wide swings cluster together → volatility clustering",
            transform=ax_ret.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFF6FF",
                      edgecolor="#BFDBFE", alpha=0.9),
        )

        # ── bottom panel: conditional vol ─────────────────────────────────────
        cv = df["conditional_vol"]
        ax_vol.fill_between(cv.index, cv, alpha=0.18, color=color)
        ax_vol.plot(cv.index, cv, color=color, linewidth=1.5,
                    label=f"{vol_model}({p},{q})-{dist.upper()} conditional vol")
        ax_vol.set_ylabel("Annualised Vol (%)", fontsize=11)
        ax_vol.set_xlabel("Date", fontsize=11)
        ax_vol.spines[["top", "right"]].set_visible(False)
        ax_vol.tick_params(labelsize=9)
        ax_vol.legend(fontsize=9, frameon=False, loc="upper right")

        ax_vol.text(
            0.01, 0.97,
            f"Persistence α+β = {persistence}\n"
            f"{'> 0.95 → vol decays slowly after spikes (long memory)' if persistence > 0.95 else 'moderate persistence'}",
            transform=ax_vol.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#EAF3DE",
                      edgecolor="#C0DD97", alpha=0.9),
        )

        ax_vol.xaxis.set_major_locator(mdates.YearLocator())
        ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # caption
        fig.text(
            0.5, -0.03,
            "The GARCH conditional vol (bottom) rises sharply when large return swings appear (top) "
            "and decays gradually during calm periods.\n"
            "High persistence means a volatility shock today is still partially present several weeks later — "
            "a core property GARCH is designed to capture.",
            ha="center", fontsize=8.5, color="#555555", style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F9F9F9",
                      edgecolor="#DDDDDD", alpha=0.95),
        )

        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"day12_returns_garch_vol_{ticker}.png")
        plt.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"  ✓  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Realised Volatility vs GARCH Forecast
# ─────────────────────────────────────────────────────────────────────────────
def plot_realised_vs_forecast(
    df       : pd.DataFrame,
    ticker   : str,
    best_row : pd.Series,
) -> None:
    """
    Single panel per ticker:
      Red shaded area  → 20-day rolling realised volatility (actual)
      Coloured line    → GARCH conditional volatility (forecast)

    What this shows:
      How well GARCH tracks actual realised volatility.
      Where GARCH undershoots (misses sudden spikes) vs overshoots.
      The RMSE and MAE on the chart are your headline GARCH baseline numbers —
      XGBoost and HARNet must beat these in Week 5+.
    """
    color       = PALETTE[ticker]
    vol_model   = best_row["vol_model"]
    p           = int(best_row["p"])
    q           = int(best_row["q"])
    dist        = best_row["dist"]
    persistence = round(best_row["persistence"], 4)

    # align both series
    plot_df = df[["realised_vol", "conditional_vol"]].dropna()
    rv = plot_df["realised_vol"]
    cv = plot_df["conditional_vol"]

    # metrics
    rmse = np.sqrt(np.mean((rv - cv) ** 2))
    mae  = np.mean(np.abs(rv - cv))
    corr = rv.corr(cv)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(16, 6))

        fig.suptitle(
            f"{ticker}  —  Realised Volatility vs "
            f"{vol_model}({p},{q})-{dist.upper()} Conditional Volatility\n"
            f"Correlation = {corr:.3f}  |  RMSE = {rmse:.4f}%  |  MAE = {mae:.4f}%  |  "
            f"Persistence = {persistence}",
            fontsize=13, fontweight="bold",
        )

        # realised vol — red shaded
        ax.fill_between(rv.index, rv, alpha=0.15, color="#d62728")
        ax.plot(rv.index, rv, color="#d62728", linewidth=1.2, alpha=0.9,
                label="Realised vol (20-day rolling, annualised)")

        # GARCH conditional vol — ticker colour
        ax.plot(cv.index, cv, color=color, linewidth=1.8, alpha=0.95,
                label=f"{vol_model}({p},{q})-{dist.upper()} conditional vol")

        # annotate biggest underestimate
        diff      = rv - cv
        worst_idx = diff.idxmax()
        ax.annotate(
            "GARCH underestimates\nspike here\n(common during\nsudden crises)",
            xy=(worst_idx, cv[worst_idx]),
            xytext=(worst_idx, cv[worst_idx] + rv.std() * 1.5),
            fontsize=8, color="#791F1F",
            arrowprops=dict(arrowstyle="->", color="#791F1F", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FCEBEB",
                      edgecolor="#F7C1C1", alpha=0.9),
        )

        # stats box — top left
        ax.text(
            0.01, 0.97,
            f"Correlation : {corr:.3f}\n"
            f"RMSE        : {rmse:.4f} %\n"
            f"MAE         : {mae:.4f} %\n"
            f"Persistence : {persistence}",
            transform=ax.transAxes, fontsize=9, va="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F9F9F9",
                      edgecolor="#CCCCCC", alpha=0.95),
        )

        ax.set_ylabel("Annualised Volatility (%)", fontsize=11)
        ax.set_xlabel("Date", fontsize=11)
        ax.legend(fontsize=10, frameon=False, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # caption
        fig.text(
            0.5, -0.04,
            "Red = actual realised volatility.  Coloured line = GARCH conditional volatility forecast.\n"
            "When lines are close → GARCH is forecasting well.  "
            "When GARCH line is below red → it underestimated a volatility spike.\n"
            "RMSE and MAE above are your GARCH baseline numbers. "
            "XGBoost and HARNet must beat these in Weeks 5–6.",
            ha="center", fontsize=8.5, color="#555555", style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F9F9F9",
                      edgecolor="#DDDDDD", alpha=0.95),
        )

        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"day12_realised_vs_forecast_{ticker}.png")
        plt.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"  ✓  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 58)
    print("  VOLATILITY FORECASTING — DAYS 12-13")
    print("  Returns + GARCH Vol  |  Realised vs Forecast")
    print("=" * 58)

    # ── step 1: load processed data using your existing loader ────────────────
    print("\n[1/3]  Loading processed data...")
    dfs = load_ticker_frames(TICKERS)
    for t, df in dfs.items():
        print(f"  ✓  {t}: {len(df):,} rows")

    # ── step 2: run GARCH tuning — same call as your Day 4 pipeline ──────────
    # This re-runs tune_garch on your real processed data.
    # Uses the same GARCH(1,1) / EGARCH, normal/t/skewt grid as before.
    print("\n[2/3]  Fitting GARCH models (reusing tune_garch_all_tickers)...")
    garch_results = tune_garch_all_tickers(dfs)

    # ── step 3: generate plots for each ticker ────────────────────────────────
    print("\n[3/3]  Generating plots...")
    for ticker in TICKERS:
        print(f"\n  ── {ticker} ──")
        result   = garch_results[ticker]
        best_row = result["best_row"]

        if best_row is None:
            print(f"  ✗  GARCH tuning failed for {ticker}, skipping.")
            continue

        best_result = result["best_result"]   # arch ARCHModelResult object
        returns     = dfs[ticker]["log_return"]

        # extract conditional vol and realised vol series
        df_vol = extract_vol_series(best_result, returns)

        # Plot 1: returns + GARCH vol
        plot_returns_and_garch_vol(df_vol, ticker, best_row)

        # Plot 2: realised vs forecast
        plot_realised_vs_forecast(df_vol, ticker, best_row)

    print("\n" + "=" * 58)
    print("  Done. Files saved to:")
    print(f"  {OUT_DIR}")
    print("─" * 58)
    for t in TICKERS:
        print(f"  day12_returns_garch_vol_{t}.png")
        print(f"  day12_realised_vs_forecast_{t}.png")
    print("─" * 58)
    print("  Next → Days 14-16: Gaussian vs t distribution comparison")
    print("=" * 58)


if __name__ == "__main__":
    main()