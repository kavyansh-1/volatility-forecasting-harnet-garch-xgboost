# ─────────────────────────────────────────────────────────────
# day11_plots.py
# All Day 11 visualisations:
#   1. Rolling vs EWMA pairwise correlation overlay
#   2. DCC dynamic correlation vs static full-sample correlation
#   3. Portfolio volatility: CCC vs DCC vs realized portfolio |return|
#   4. Diversification benefit over time
#   5. Portfolio VaR backtest (CCC vs DCC, with violation markers)
#   6. Correlation matrix heatmap snapshot: calm day vs stressed day
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

TICKERS  = ["SPY", "QQQ", "AAPL"]
PAIRS    = [("SPY","QQQ"), ("SPY","AAPL"), ("QQQ","AAPL")]
PAIR_COLORS = {
    "SPY_QQQ" : "#1f77b4",
    "SPY_AAPL": "#ff7f0e",
    "QQQ_AAPL": "#2ca02c",
}


# ── Plot 1: Rolling vs EWMA correlation ─────────────────────────
def plot_rolling_vs_ewma(roll_corr_df: pd.DataFrame,
                          ewma_corr_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(PAIRS), 1, figsize=(13, 3.3*len(PAIRS)))
    fig.suptitle("Pairwise Correlation — Rolling (21d) vs EWMA (λ=0.94)",
                 fontsize=13, fontweight="bold")

    for ax, (a, b) in zip(axes, PAIRS):
        col   = f"corr_{a}_{b}"
        color = PAIR_COLORS[f"{a}_{b}"]

        ax.plot(roll_corr_df.index, roll_corr_df[col],
                color=color, linewidth=1.0, alpha=0.55,
                label="Rolling 21d")
        ax.plot(ewma_corr_df.index, ewma_corr_df[col],
                color=color, linewidth=1.4, alpha=0.95,
                linestyle="--", label="EWMA")

        ax.axhline(0, color="black", linewidth=0.6,
                   linestyle=":", alpha=0.4)
        ax.set_title(f"{a} – {b}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Correlation")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day11_rolling_vs_ewma_correlation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 2: DCC vs static correlation ───────────────────────────
def plot_dcc_vs_static(dcc_corr_df: pd.DataFrame,
                        full_corr:   np.ndarray) -> None:
    idx_map = {t: i for i, t in enumerate(TICKERS)}

    fig, axes = plt.subplots(len(PAIRS), 1, figsize=(13, 3.3*len(PAIRS)))
    fig.suptitle("DCC Dynamic Correlation vs Static Full-Sample Correlation",
                 fontsize=13, fontweight="bold")

    for ax, (a, b) in zip(axes, PAIRS):
        col    = f"dcc_corr_{a}_{b}"
        static = full_corr[idx_map[a], idx_map[b]]
        color  = PAIR_COLORS[f"{a}_{b}"]

        ax.plot(dcc_corr_df.index, dcc_corr_df[col],
                color=color, linewidth=1.1, alpha=0.85,
                label="DCC (dynamic)")
        ax.axhline(static, color="black", linewidth=1.4,
                   linestyle="--",
                   label=f"Static avg = {static:.3f}")

        ax.set_title(f"{a} – {b}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Correlation")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day11_dcc_vs_static_correlation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 3: Portfolio vol CCC vs DCC ────────────────────────────
def plot_portfolio_vol_comparison(port_results: dict) -> None:
    port_ret     = port_results["port_ret"]
    port_vol_ccc = port_results["port_vol_ccc"]
    port_vol_dcc = port_results["port_vol_dcc"]

    common = port_vol_dcc.index.intersection(port_ret.index)
    realized_abs_ret = port_ret.loc[common].abs()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(common, realized_abs_ret * 100, color="grey",
            linewidth=0.7, alpha=0.5, label="|Realised portfolio return|")
    ax.plot(common, port_vol_ccc.loc[common] * 100, color="#7f7f7f",
            linewidth=1.3, label="CCC portfolio vol forecast")
    ax.plot(common, port_vol_dcc.loc[common] * 100, color="#9467bd",
            linewidth=1.3, label="DCC portfolio vol forecast")

    ax.set_title("Portfolio Volatility Forecast — CCC vs DCC",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Daily Vol (%)")
    ax.set_xlabel("Date")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day11_portfolio_vol_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 4: Diversification benefit ─────────────────────────────
def plot_diversification_benefit(div_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle("Diversification Benefit Over Time",
                 fontsize=13, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(div_df.index, div_df["weighted_sum_vol"] * 100,
            color="#d62728", linewidth=1.1, alpha=0.8,
            label="Weighted sum of individual vols\n(no diversification)")
    ax1.plot(div_df.index, div_df["portfolio_vol_dcc"] * 100,
            color="#9467bd", linewidth=1.2, alpha=0.9,
            label="Actual portfolio vol (DCC)")
    ax1.fill_between(div_df.index,
                     div_df["portfolio_vol_dcc"] * 100,
                     div_df["weighted_sum_vol"] * 100,
                     color="green", alpha=0.12)
    ax1.set_ylabel("Daily Vol (%)")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(True, alpha=0.2, linestyle="--")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    ax2.fill_between(div_df.index,
                     div_df["diversification_benefit"] * 100,
                     color="green", alpha=0.35)
    ax2.plot(div_df.index, div_df["diversification_benefit"] * 100,
            color="darkgreen", linewidth=0.8)
    ax2.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax2.set_ylabel("Diversification\nbenefit (% vol)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.2, linestyle="--")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day11_diversification_benefit.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 5: Portfolio VaR backtest ──────────────────────────────
def plot_portfolio_var_backtest(series_store: dict,
                                 alpha: float = 0.05) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle(f"Portfolio VaR Backtest at {int((1-alpha)*100)}% "
                 f"— CCC vs DCC",
                 fontsize=13, fontweight="bold")

    for ax, label, color in zip(axes, ["CCC", "DCC"],
                                ["#7f7f7f", "#9467bd"]):
        s = series_store[(label, alpha)]
        ax.plot(s["dates"], s["actual"] * 100,
                color=color, linewidth=0.7, alpha=0.6,
                label="Actual portfolio return")
        ax.plot(s["dates"], s["VaR"] * 100,
                color="red", linewidth=1.2, linestyle="--",
                label="VaR threshold")

        viol = s["violations"].astype(bool)
        ax.scatter(
            np.array(s["dates"])[viol],
            s["actual"][viol] * 100,
            color="darkred", s=25, zorder=5,
            label=f"Violations (n={viol.sum()})"
        )
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("Daily Return (%)")
        ax.legend(frameon=False, fontsize=8, ncol=3)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    out = os.path.join(
        OUT_DIR, f"day11_portfolio_var_backtest_{int(alpha*100)}.png"
    )
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 6: Correlation heatmap snapshot ────────────────────────
def plot_correlation_heatmap_snapshot(dcc_corr_df: pd.DataFrame,
                                       returns: pd.DataFrame) -> None:
    """
    Compares the DCC correlation matrix on the calmest day
    (lowest realised |return| sum) vs the most stressed day
    (highest realised |return| sum) in the sample.
    """
    idx_map = {t: i for i, t in enumerate(TICKERS)}
    abs_ret_sum = returns.abs().sum(axis=1)
    common = dcc_corr_df.index.intersection(abs_ret_sum.index)

    calm_date    = abs_ret_sum.loc[common].idxmin()
    stress_date  = abs_ret_sum.loc[common].idxmax()

    def build_matrix(date):
        row = dcc_corr_df.loc[date]
        mat = np.eye(len(TICKERS))
        for a, b in PAIRS:
            v = row[f"dcc_corr_{a}_{b}"]
            mat[idx_map[a], idx_map[b]] = v
            mat[idx_map[b], idx_map[a]] = v
        return mat

    calm_mat   = build_matrix(calm_date)
    stress_mat = build_matrix(stress_date)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("DCC Correlation Matrix — Calm vs Stressed Day",
                 fontsize=13, fontweight="bold")

    for ax, mat, date, title in zip(
        axes, [calm_mat, stress_mat], [calm_date, stress_date],
        ["Calmest day", "Most stressed day"]
    ):
        sns.heatmap(
            pd.DataFrame(mat, index=TICKERS, columns=TICKERS),
            ax=ax, annot=True, fmt=".3f",
            cmap="RdYlGn_r", vmin=-1, vmax=1,
            linewidths=0.5, cbar_kws={"label": "Correlation"},
        )
        ax.set_title(f"{title}\n{pd.Timestamp(date).date()}",
                     fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day11_correlation_heatmap_snapshot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Main ────────────────────────────────────────────────────────
def run_all_plots(cov_results:  dict,
                  dcc_results:  dict,
                  port_results: dict) -> None:
    print(f"\n{'='*55}")
    print("  DAY 11 — Generating Plots")
    print(f"{'='*55}")

    plot_rolling_vs_ewma(cov_results["roll_corr_df"],
                         cov_results["ewma_corr_df"])
    plot_dcc_vs_static(dcc_results["dcc_corr_df"],
                       port_results["full_corr"])
    plot_portfolio_vol_comparison(port_results)
    plot_diversification_benefit(port_results["div_df"])
    plot_portfolio_var_backtest(port_results["series_store"], alpha=0.05)
    plot_portfolio_var_backtest(port_results["series_store"], alpha=0.01)
    plot_correlation_heatmap_snapshot(dcc_results["dcc_corr_df"],
                                      cov_results["returns"])

    print("\n  All Day 11 plots complete.")