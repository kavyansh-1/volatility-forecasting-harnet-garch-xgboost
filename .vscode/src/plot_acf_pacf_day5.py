"""
src/plot_acf_pacf.py
--------------------
Day 5-7 Task: Generate clean, labeled ACF and PACF plots
for log returns and squared log returns for all assets.

Saves PNGs to:  plots/day5/
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# ── Style constants (consistent across all plots) ────────────────────────────
STYLE       = "seaborn-v0_8-whitegrid"
FIGSIZE_ACF = (16, 10)      # width x height for the 2×2 ACF/PACF panel
DPI         = 150
NLAGS       = 40            # how many lags to show

COLOR_ACF   = "#1f77b4"     # blue  — ACF bars
COLOR_PACF  = "#ff7f0e"     # orange — PACF bars
COLOR_CONF  = "#d62728"     # red   — confidence interval lines
ALPHA_BAR   = 0.75

TICKER_COLORS = {
    "SPY" : "#1f77b4",
    "QQQ" : "#ff7f0e",
    "AAPL": "#2ca02c",
}


# ── Helper: load processed CSV ────────────────────────────────────────────────
def load_returns(ticker: str, data_dir: str = "data/processed") -> pd.Series:
    """Load log_return series for one ticker from processed CSV."""
    path = os.path.join(data_dir, f"{ticker}_processed.csv")
    df   = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df["log_return"].dropna()


# ── Core plot: ACF + PACF panel for one series ───────────────────────────────
def _acf_pacf_panel(
    ax_acf:  plt.Axes,
    ax_pacf: plt.Axes,
    series:  pd.Series,
    title:   str,
    color:   str,
    nlags:   int = NLAGS,
) -> None:
    """
    Draw ACF on ax_acf and PACF on ax_pacf.
    Called twice per asset — once for returns, once for squared returns.
    """
    # ── ACF ──────────────────────────────────────────────────────────────────
    plot_acf(
        series,
        ax=ax_acf,
        lags=nlags,
        alpha=0.05,             # 95% confidence band
        color=color,
        vlines_kwargs={"linewidth": 1.2},
        title="",               # we set our own title below
        zero=False,             # skip lag-0 (always 1.0, uninformative)
        auto_ylims=True,
    )
    ax_acf.set_title(f"ACF — {title}", fontsize=11, fontweight="bold", pad=8)
    ax_acf.set_xlabel("Lag (days)", fontsize=10)
    ax_acf.set_ylabel("Autocorrelation", fontsize=10)
    ax_acf.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax_acf.spines[["top", "right"]].set_visible(False)
    ax_acf.tick_params(labelsize=9)

    # ── PACF ─────────────────────────────────────────────────────────────────
    plot_pacf(
        series,
        ax=ax_pacf,
        lags=nlags,
        alpha=0.05,
        method="ywm",           # Yule-Walker — most stable for financial data
        color=color,
        vlines_kwargs={"linewidth": 1.2},
        title="",
        zero=False,
        auto_ylims=True,
    )
    ax_pacf.set_title(f"PACF — {title}", fontsize=11, fontweight="bold", pad=8)
    ax_pacf.set_xlabel("Lag (days)", fontsize=10)
    ax_pacf.set_ylabel("Partial Autocorrelation", fontsize=10)
    ax_pacf.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax_pacf.spines[["top", "right"]].set_visible(False)
    ax_pacf.tick_params(labelsize=9)


# ── Plot 1: Individual asset — returns vs squared returns (2×2 panel) ────────
def plot_acf_pacf_single(
    returns:  pd.Series,
    ticker:   str,
    save_dir: str = "plots/day5",
    nlags:    int = NLAGS,
) -> None:
    """
    Produces a 2×2 panel for one ticker:
      Top row    → ACF and PACF of log returns
      Bottom row → ACF and PACF of squared log returns

    Why squared returns?
      Squared returns proxy realized variance. If they show
      autocorrelation, volatility clusters — validating GARCH.
    """
    os.makedirs(save_dir, exist_ok=True)
    squared = returns ** 2
    color   = TICKER_COLORS.get(ticker, "#333333")

    with plt.style.context(STYLE):
        fig = plt.figure(figsize=FIGSIZE_ACF)
        fig.suptitle(
            f"{ticker}  —  ACF / PACF of Returns and Squared Returns\n"
            f"(lags = {nlags}  |  shaded band = 95% confidence interval)",
            fontsize=13, fontweight="bold", y=1.01,
        )

        gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32)
        ax_r_acf   = fig.add_subplot(gs[0, 0])
        ax_r_pacf  = fig.add_subplot(gs[0, 1])
        ax_r2_acf  = fig.add_subplot(gs[1, 0])
        ax_r2_pacf = fig.add_subplot(gs[1, 1])

        _acf_pacf_panel(ax_r_acf,  ax_r_pacf,
                        returns, f"{ticker} Log Returns",        color, nlags)
        _acf_pacf_panel(ax_r2_acf, ax_r2_pacf,
                        squared, f"{ticker} Squared Returns",    color, nlags)

        # ── Annotation box explaining what to look for ───────────────────────
        note = (
            "Top row: Log returns — expect NO significant spikes (efficient markets).\n"
            "Bottom row: Squared returns — expect MANY spikes (volatility clustering → GARCH justified)."
        )
        fig.text(
            0.5, -0.02, note,
            ha="center", va="top", fontsize=9,
            color="#555555", style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f9f9f9",
                      edgecolor="#cccccc", alpha=0.9),
        )

        plt.tight_layout()
        out = os.path.join(save_dir, f"acf_pacf_{ticker}.png")
        plt.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {out}")


# ── Plot 2: All-asset comparison — one row per asset (3×4 mega panel) ────────
def plot_acf_pacf_all_assets(
    returns_dict: dict,
    save_dir: str = "plots/day5",
    nlags:    int = NLAGS,
) -> None:
    """
    One big figure: 3 rows (SPY / QQQ / AAPL), 4 columns:
      Col 1: ACF of returns
      Col 2: PACF of returns
      Col 3: ACF of squared returns
      Col 4: PACF of squared returns

    Lets you compare all three assets side-by-side at a glance.
    """
    os.makedirs(save_dir, exist_ok=True)
    tickers = list(returns_dict.keys())
    n       = len(tickers)

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(
            n, 4,
            figsize=(22, 4.5 * n),
            gridspec_kw={"hspace": 0.55, "wspace": 0.30},
        )
        if n == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(
            "ACF / PACF Comparison — All Assets\n"
            "Columns: ACF(returns) | PACF(returns) | ACF(r²) | PACF(r²)",
            fontsize=14, fontweight="bold", y=1.01,
        )

        col_titles = [
            "ACF — Log Returns",
            "PACF — Log Returns",
            "ACF — Squared Returns",
            "PACF — Squared Returns",
        ]

        for row_i, ticker in enumerate(tickers):
            r      = returns_dict[ticker]
            r2     = r ** 2
            color  = TICKER_COLORS.get(ticker, "#333333")

            # row label on the left
            axes[row_i, 0].set_ylabel(
                ticker, fontsize=14, fontweight="bold",
                color=color, labelpad=14, rotation=0, va="center",
            )

            for col_i, (series, is_pacf) in enumerate(
                [(r, False), (r, True), (r2, False), (r2, True)]
            ):
                ax = axes[row_i, col_i]

                if is_pacf:
                    plot_pacf(series, ax=ax, lags=nlags, alpha=0.05,
                              method="ywm", color=color,
                              vlines_kwargs={"linewidth": 1.0},
                              title="", zero=False, auto_ylims=True)
                else:
                    plot_acf(series, ax=ax, lags=nlags, alpha=0.05,
                             color=color,
                             vlines_kwargs={"linewidth": 1.0},
                             title="", zero=False, auto_ylims=True)

                if row_i == 0:
                    ax.set_title(col_titles[col_i],
                                 fontsize=10, fontweight="bold", pad=6)
                ax.set_xlabel("Lag", fontsize=9)
                ax.axhline(0, color="black", linewidth=0.6)
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(labelsize=8)

        plt.tight_layout()
        out = os.path.join(save_dir, "acf_pacf_all_assets_comparison.png")
        plt.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {out}")


# ── Plot 3: Ljung-Box p-value chart ──────────────────────────────────────────
def plot_ljung_box_pvalues(
    returns_dict: dict,
    save_dir: str = "plots/day5",
    max_lag: int  = 20,
) -> None:
    """
    Plots Ljung-Box p-values at each lag for returns and squared returns.

    Why: If p-value < 0.05 at many lags → significant autocorrelation.
    For squared returns this confirms volatility clustering → GARCH is justified.
    For raw returns this tests efficient market hypothesis.
    """
    os.makedirs(save_dir, exist_ok=True)
    lags    = list(range(1, max_lag + 1))
    tickers = list(returns_dict.keys())

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(
            1, len(tickers),
            figsize=(7 * len(tickers), 5),
            sharey=True,
        )
        if len(tickers) == 1:
            axes = [axes]

        fig.suptitle(
            "Ljung-Box Test P-values by Lag\n"
            "(Below red dashed line = significant autocorrelation at 5% level)",
            fontsize=13, fontweight="bold",
        )

        for ax, ticker in zip(axes, tickers):
            r  = returns_dict[ticker]
            r2 = r ** 2
            color = TICKER_COLORS.get(ticker, "#333333")

            lb_r  = acorr_ljungbox(r,  lags=lags, return_df=True)
            lb_r2 = acorr_ljungbox(r2, lags=lags, return_df=True)

            ax.plot(lags, lb_r["lb_pvalue"],
                    marker="o", markersize=4, linewidth=1.5,
                    color=color, label="Log returns", alpha=0.9)
            ax.plot(lags, lb_r2["lb_pvalue"],
                    marker="s", markersize=4, linewidth=1.5,
                    color="#d62728", label="Squared returns",
                    linestyle="--", alpha=0.9)

            ax.axhline(0.05, color="red", linewidth=1.2,
                       linestyle="--", alpha=0.7, label="5% threshold")
            ax.fill_between(lags, 0, 0.05,
                            color="red", alpha=0.05)

            ax.set_title(f"{ticker}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Lag (days)", fontsize=10)
            ax.set_ylabel("p-value" if ticker == tickers[0] else "",
                          fontsize=10)
            ax.set_ylim(-0.02, 1.05)
            ax.legend(fontsize=9, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=9)

        plt.tight_layout()
        out = os.path.join(save_dir, "ljung_box_pvalues.png")
        plt.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {out}")


# ── EOD_DAY5.py runner (paste this in your root EOD_DAY5.py) ─────────────────
def run_day5(
    tickers:  list = None,
    data_dir: str  = "data/processed",
    save_dir: str  = "plots/day5",
    nlags:    int  = NLAGS,
) -> None:
    """
    Main runner — call this from EOD_DAY5.py.
    Loads returns for each ticker and generates all 3 plot types.
    """
    if tickers is None:
        tickers = ["SPY", "QQQ", "AAPL"]

    print("\n" + "=" * 58)
    print("  VOLATILITY FORECASTING PROJECT — DAY 5")
    print("  ACF / PACF Diagnostic Plots")
    print("=" * 58)

    # ── Load returns ─────────────────────────────────────────────────────────
    print("\n[1/4]  Loading processed returns...")
    returns_dict = {}
    for ticker in tickers:
        try:
            returns_dict[ticker] = load_returns(ticker, data_dir)
            n = len(returns_dict[ticker])
            print(f"  ✓  {ticker}: {n:,} observations")
        except FileNotFoundError:
            print(f"  ✗  {ticker}: processed CSV not found in {data_dir}")

    if not returns_dict:
        print("  ERROR: No data loaded. Run EOD_DAY2.py first.")
        return

    # ── Individual asset plots ────────────────────────────────────────────────
    print(f"\n[2/4]  Generating individual ACF/PACF panels (one per asset)...")
    for ticker, r in returns_dict.items():
        plot_acf_pacf_single(r, ticker, save_dir=save_dir, nlags=nlags)

    # ── All-asset comparison panel ────────────────────────────────────────────
    print(f"\n[3/4]  Generating all-asset comparison panel...")
    plot_acf_pacf_all_assets(returns_dict, save_dir=save_dir, nlags=nlags)

    # ── Ljung-Box p-value chart ───────────────────────────────────────────────
    print(f"\n[4/4]  Generating Ljung-Box p-value charts...")
    plot_ljung_box_pvalues(returns_dict, save_dir=save_dir)

    print("\n" + "=" * 58)
    print("  Day 5 complete.")
    print("─" * 58)
    print(f"  plots/day5/   — {2*len(returns_dict) + 2} new PNG files")
    print("─" * 58)
    print("\n  Files generated:")
    for ticker in returns_dict:
        print(f"    acf_pacf_{ticker}.png")
    print("    acf_pacf_all_assets_comparison.png")
    print("    ljung_box_pvalues.png")
    print("=" * 58)


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_day5()
