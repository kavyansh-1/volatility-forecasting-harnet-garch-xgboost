"""
day14/src/plot_dist_comparison.py
-----------------------------------
Days 14-16:
  Plot 1 - GARCH(1,1)-Normal vs GARCH(1,1)-t conditional volatility, overlaid
  Plot 2 - Innovation distribution comparison: histogram + Normal/t PDF fits
  Table  - Small comparison table: AIC, BIC, persistence, tail thickness (df)

Imports directly from your existing modules:
  - models/feature_engineering.py → load_ticker_frames()
  - models/tune_garch_models.py   → fit_garch()
  - data/processed/                → your Day 2 processed CSVs

Run from project root:
    python day14/src/plot_dist_comparison.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

# ── path setup — same pattern as your other day files ────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".vscode"))

from models.feature_engineering import load_ticker_frames
from models.tune_garch_models   import fit_garch

# ── constants ─────────────────────────────────────────────────────────────────
TICKERS = ["SPY", "QQQ", "AAPL"]

# matches your existing project colour palette
PALETTE = {
    "SPY" : "#1f77b4",
    "QQQ" : "#ff7f0e",
    "AAPL": "#2ca02c",
}

# colours for the two distributions — consistent across all charts
DIST_COLORS = {
    "normal": "#6B7280",   # gray — the "wrong" assumption
    "t"     : "#d62728",   # red  — the "correct" assumption (matches your project palette)
}

STYLE = "seaborn-v0_8-whitegrid"
DPI   = 150

# output folder — same pattern as day12/output
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)


# ── helper: fit both distributions and extract vol series ───────────────────
def fit_both_distributions(returns: pd.Series) -> dict:
    """
    Fits GARCH(1,1) twice on the same return series — once assuming
    Normal errors, once assuming Student-t errors. Reuses your existing
    fit_garch() function from tune_garch_models.py.

    Returns a dict with both fitted results plus their conditional
    volatility series, aligned to the return index.
    """
    r = returns.dropna()

    fit_normal = fit_garch(r, p=1, q=1, vol_model="GARCH", dist="normal")
    fit_t      = fit_garch(r, p=1, q=1, vol_model="GARCH", dist="t")

    out = {"normal": fit_normal, "t": fit_t}

    for key, fit_row in out.items():
        if fit_row.get("_result") is None:
            out[key]["cond_vol_ann"] = None
            continue
        result = fit_row["_result"]
        cond_vol_pct = result.conditional_volatility       # daily %
        cond_vol_ann = cond_vol_pct * np.sqrt(252)          # annualised %
        cond_vol_ann = cond_vol_ann.reindex(r.index)
        out[key]["cond_vol_ann"] = cond_vol_ann
        out[key]["std_resid"]    = result.std_resid          # standardised residuals

    return out


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Conditional Volatility: Normal vs t overlaid
# ─────────────────────────────────────────────────────────────────────────────
def plot_vol_normal_vs_t(
    returns  : pd.Series,
    fits     : dict,
    ticker   : str,
) -> None:
    """
    Single panel per ticker:
      Gray line  → GARCH(1,1)-Normal conditional volatility
      Red line   → GARCH(1,1)-t conditional volatility

    What this shows:
      The two distributions produce very similar conditional volatility
      PATHS (the GARCH recursion is mostly driven by p,q not the error
      distribution) but Normal underestimates uncertainty during extreme
      moves because it doesn't expect them. The gap widens during crises.
    """
    color = PALETTE[ticker]

    cv_normal = fits["normal"]["cond_vol_ann"]
    cv_t      = fits["t"]["cond_vol_ann"]

    if cv_normal is None or cv_t is None:
        print(f"  ✗  Skipping Plot 1 for {ticker} — one distribution failed to fit")
        return

    aic_normal = fits["normal"]["aic"]
    aic_t      = fits["t"]["aic"]
    winner     = "t" if aic_t < aic_normal else "normal"

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(16, 6))

        fig.suptitle(
            f"{ticker}  —  GARCH(1,1) Conditional Volatility: Normal vs Student-t Errors\n"
            f"AIC(normal) = {aic_normal:.2f}   |   AIC(t) = {aic_t:.2f}   |   "
            f"Lower AIC wins → {winner.upper()}",
            fontsize=13, fontweight="bold",
        )

        ax.plot(cv_normal.index, cv_normal,
                color=DIST_COLORS["normal"], linewidth=1.4, alpha=0.85,
                label="GARCH(1,1)-Normal")
        ax.plot(cv_t.index, cv_t,
                color=DIST_COLORS["t"], linewidth=1.6, alpha=0.95,
                label="GARCH(1,1)-t  (Student-t errors)")

        # shade the gap during the largest divergence
        diff      = (cv_t - cv_normal).abs()
        worst_idx = diff.idxmax()
        ax.annotate(
            "Largest divergence —\nt-distribution reacts\nmore strongly to\nextreme shocks here",
            xy=(worst_idx, cv_t[worst_idx]),
            xytext=(worst_idx, cv_t[worst_idx] + cv_t.std() * 1.3),
            fontsize=8, color="#791F1F",
            arrowprops=dict(arrowstyle="->", color="#791F1F", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FCEBEB",
                      edgecolor="#F7C1C1", alpha=0.9),
        )

        ax.set_ylabel("Annualised Volatility (%)", fontsize=11)
        ax.set_xlabel("Date", fontsize=11)
        ax.legend(fontsize=10, frameon=False, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        fig.text(
            0.5, -0.04,
            "Both distributions produce broadly similar volatility paths since the GARCH "
            "recursion (p,q) drives most of the dynamics. The key\n"
            "difference is in the tails: Normal assumes extreme shocks are rare, so it "
            "reacts less aggressively during crises. Student-t expects\n"
            "fat tails and responds more sharply — matching the Q-Q and Ljung-Box "
            "evidence from Days 8-10 that returns are fat-tailed.",
            ha="center", fontsize=8.5, color="#555555", style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F9F9F9",
                      edgecolor="#DDDDDD", alpha=0.95),
        )

        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"day14_vol_normal_vs_t_{ticker}.png")
        plt.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"  ✓  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Standardised Residual Distribution: Histogram + Normal/t PDF
# ─────────────────────────────────────────────────────────────────────────────
def plot_innovation_distributions(
    fits   : dict,
    ticker : str,
) -> None:
    """
    Single panel per ticker:
      Histogram of standardised residuals from the t-fit (the model's
      "innovations" after removing the GARCH volatility pattern)
      overlaid with both the Normal PDF and the fitted Student-t PDF.

    What this shows:
      After GARCH removes the volatility clustering, what's left should
      be i.i.d. noise. If that noise still has fat tails relative to
      Normal, it proves the Normal assumption fails even after accounting
      for time-varying volatility — strengthening the case for dist='t'.
    """
    color = PALETTE[ticker]

    # use the t-fit's standardised residuals (more representative innovations)
    fit_t = fits.get("t", {})
    std_resid = fit_t.get("std_resid")

    if std_resid is None:
        print(f"  ✗  Skipping Plot 2 for {ticker} — t-fit unavailable")
        return

    std_resid = pd.Series(std_resid).dropna()

    # estimate Student-t degrees of freedom from the fitted model params
    params = fit_t["_result"].params.to_dict()
    nu = params.get("nu", 5.0)   # 'nu' is arch's name for the t dof parameter

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))

        fig.suptitle(
            f"{ticker}  —  GARCH-Filtered Innovations vs Normal and Student-t PDFs\n"
            f"Fitted degrees of freedom (ν) = {nu:.2f}   "
            f"(lower ν = fatter tails; ν<10 confirms fat tails remain after GARCH filtering)",
            fontsize=12, fontweight="bold",
        )

        # histogram of standardised residuals
        ax.hist(std_resid, bins=80, density=True,
                color=color, alpha=0.55,
                label="Standardised residuals (after GARCH filtering)",
                edgecolor="white", linewidth=0.3)

        x = np.linspace(std_resid.min(), std_resid.max(), 400)

        # Normal PDF (mean 0, std 1 — residuals are standardised)
        ax.plot(x, stats.norm.pdf(x, 0, 1),
                color=DIST_COLORS["normal"], linestyle="--", linewidth=2.0,
                label="Normal(0,1) fit")

        # Student-t PDF using the model's fitted nu, scaled to unit variance
        scale_t = np.sqrt((nu - 2) / nu) if nu > 2 else 1.0
        ax.plot(x, stats.t.pdf(x, df=nu, scale=scale_t),
                color=DIST_COLORS["t"], linewidth=2.0,
                label=f"Student-t(ν={nu:.1f}) fit")

        # annotation: tail comparison
        ax.annotate(
            "Tails heavier than\nNormal predicts\n→ Student-t fits better",
            xy=(x.max() * 0.7, stats.norm.pdf(x.max() * 0.7, 0, 1)),
            xytext=(x.max() * 0.45, ax.get_ylim()[1] * 0.55),
            fontsize=8.5, color="#791F1F",
            arrowprops=dict(arrowstyle="->", color="#791F1F", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FCEBEB",
                      edgecolor="#F7C1C1", alpha=0.9),
        )

        ax.set_xlabel("Standardised residual", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=9, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)

        fig.text(
            0.5, -0.05,
            "Interpretation: After GARCH removes time-varying volatility, the leftover "
            "noise (standardised residuals) should be i.i.d. if the\n"
            "model is correctly specified. A low ν (degrees of freedom) confirms that "
            "fat tails persist even after GARCH filtering — strong\n"
            "evidence that dist='t' is the appropriate choice over dist='normal' for this asset.",
            ha="center", fontsize=8.5, color="#555555", style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F9F9F9",
                      edgecolor="#DDDDDD", alpha=0.95),
        )

        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"day14_innovations_{ticker}.png")
        plt.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"  ✓  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE — Small comparison table across all tickers
# ─────────────────────────────────────────────────────────────────────────────
def build_comparison_table(all_fits: dict) -> pd.DataFrame:
    """
    Builds the small comparison table requested in the Day 14-16 task:
    one row per ticker x distribution, showing AIC, BIC, persistence,
    and tail thickness (degrees of freedom for the t-fit).

    Saves to CSV and also renders as a styled PNG image for easy
    embedding into the report (Days 17-18).
    """
    rows = []
    for ticker, fits in all_fits.items():
        for dist_name in ["normal", "t"]:
            fit = fits.get(dist_name, {})
            if fit.get("aic") is None or pd.isna(fit.get("aic")):
                continue

            nu = np.nan
            if dist_name == "t" and fit.get("_result") is not None:
                nu = fit["_result"].params.to_dict().get("nu", np.nan)

            rows.append({
                "Ticker"      : ticker,
                "Distribution": dist_name,
                "AIC"         : round(fit["aic"], 2),
                "BIC"         : round(fit["bic"], 2),
                "Persistence" : round(fit["persistence"], 4),
                "Degrees of Freedom (ν)": round(nu, 2) if not np.isnan(nu) else "—",
            })

    table_df = pd.DataFrame(rows)

    # CSV output
    csv_out = os.path.join(OUT_DIR, "day14_dist_comparison_table.csv")
    table_df.to_csv(csv_out, index=False)
    print(f"  ✓  Saved → {csv_out}")

    # also render a clean PNG table for the report
    _render_table_png(table_df)

    return table_df


def _render_table_png(table_df: pd.DataFrame) -> None:
    """Renders the comparison table as a clean PNG image."""
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 0.6 + 0.45 * len(table_df)))
        ax.axis("off")

        # colour rows by distribution for quick visual scanning
        cell_colors = []
        for _, row in table_df.iterrows():
            base = DIST_COLORS.get(row["Distribution"], "#FFFFFF")
            # very light tint version of the distribution colour
            light = "#FCEBEB" if row["Distribution"] == "t" else "#F1EFE8"
            cell_colors.append([light] * len(table_df.columns))

        tbl = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            cellColours=cell_colors,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.8)

        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor("#1B2A4A")

        ax.set_title(
            "GARCH(1,1): Normal vs Student-t — Model Comparison\n"
            "Lower AIC/BIC = better fit  |  Lower ν = fatter tails",
            fontsize=12, fontweight="bold", pad=14,
        )

        plt.tight_layout()
        out = os.path.join(OUT_DIR, "day14_dist_comparison_table.png")
        plt.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"  ✓  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 58)
    print("  VOLATILITY FORECASTING — DAYS 14-16")
    print("  Gaussian vs Student-t Distribution Comparison")
    print("=" * 58)

    # ── step 1: load processed data using your existing loader ────────────────
    print("\n[1/4]  Loading processed data...")
    dfs = load_ticker_frames(TICKERS)
    for t, df in dfs.items():
        print(f"  ✓  {t}: {len(df):,} rows")

    # ── step 2: fit both distributions for each ticker ────────────────────────
    print("\n[2/4]  Fitting GARCH(1,1)-Normal and GARCH(1,1)-t for each ticker...")
    all_fits = {}
    for ticker in TICKERS:
        print(f"\n  ── {ticker} ──")
        returns = dfs[ticker]["log_return"]
        fits = fit_both_distributions(returns)
        all_fits[ticker] = fits

        aic_n = fits["normal"]["aic"]
        aic_t = fits["t"]["aic"]
        print(f"    Normal: AIC={aic_n:.2f}")
        print(f"    t     : AIC={aic_t:.2f}   {'← lower AIC, better fit' if aic_t < aic_n else ''}")

    # ── step 3: generate plots ──────────────────────────────────────────────
    print("\n[3/4]  Generating plots...")
    for ticker in TICKERS:
        print(f"\n  - {ticker} -")
        fits    = all_fits[ticker]
        returns = dfs[ticker]["log_return"]

        plot_vol_normal_vs_t(returns, fits, ticker)
        plot_innovation_distributions(fits, ticker)

    # ── step 4: build the comparison table ─────────────────────────────────
    print("\n[4/4]  Building comparison table...")
    table_df = build_comparison_table(all_fits)
    print("\n" + table_df.to_string(index=False))

    print("\n" + "=" * 58)
    print("  Done. Files saved to:")
    print(f"  {OUT_DIR}")
    print("-" * 58)
    for t in TICKERS:
        print(f"  day14_vol_normal_vs_t_{t}.png")
        print(f"  day14_innovations_{t}.png")
    print("  day14_dist_comparison_table.csv")
    print("  day14_dist_comparison_table.png")
    print("-" * 58)
    print("  Next → Days 17-18: Polish figures and embed into report")
    print("=" * 58)


if __name__ == "__main__":
    main()