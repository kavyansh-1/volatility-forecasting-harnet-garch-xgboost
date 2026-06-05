# ─────────────────────────────────────────────────────────────
# day06_plots.py
# All Day 6 visualisations:
#   1. Daily sentiment compound score over time (per ticker)
#   2. Sentiment distribution histograms
#   3. Rolling 7-day sentiment vs rolling volatility overlay
#   4. Correlation heatmap: sentiment features vs RV metrics
#   5. Article count per day (coverage chart)
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

TICKER_COL = {"SPY": "#1f77b4", "QQQ": "#ff7f0e", "AAPL": "#2ca02c"}
TICKERS    = ["SPY", "QQQ", "AAPL"]


def load_merged(ticker: str) -> pd.DataFrame:
    path = os.path.join(OUT_DIR, f"day06_{ticker}_sentiment_vol.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col="Date", parse_dates=True)


# ── Plot 1: Compound sentiment over time ────────────────────────
def plot_sentiment_timeseries() -> None:
    fig, axes = plt.subplots(len(TICKERS), 1,
                              figsize=(13, 3.5 * len(TICKERS)),
                              sharex=False)
    fig.suptitle("Daily Compound Sentiment Score (FinBERT)",
                 fontsize=13, fontweight="bold")

    for ax, ticker in zip(axes, TICKERS):
        df = load_merged(ticker)
        if df is None:
            continue
        color = TICKER_COL[ticker]
        sent  = df["sent_compound_mean"]

        # Colour-coded bars: green = positive, red = negative
        pos = sent.clip(lower=0)
        neg = sent.clip(upper=0)
        ax.bar(df.index, pos, color="green", alpha=0.5,
               width=1.5, label="Positive")
        ax.bar(df.index, neg, color="red",   alpha=0.5,
               width=1.5, label="Negative")

        # 7-day rolling average
        ax.plot(df.index, df["sent_roll7"],
                color=color, linewidth=1.4,
                label="7-day MA", zorder=5)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.set_ylabel("Compound Score")
        ax.legend(frameon=False, fontsize=8, ncol=3)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day06_sentiment_timeseries.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 2: Sentiment distribution ──────────────────────────────
def plot_sentiment_distribution() -> None:
    fig, axes = plt.subplots(1, len(TICKERS),
                              figsize=(5 * len(TICKERS), 4))
    fig.suptitle("Distribution of Daily Compound Sentiment",
                 fontsize=13, fontweight="bold")

    for ax, ticker in zip(axes, TICKERS):
        df = load_merged(ticker)
        if df is None:
            continue
        color = TICKER_COL[ticker]
        sent  = df["sent_compound_mean"].dropna()

        ax.hist(sent, bins=40, color=color, alpha=0.65,
                edgecolor="none", density=True)
        ax.axvline(sent.mean(), color="black",
                   linewidth=1.5, linestyle="--",
                   label=f"Mean={sent.mean():.3f}")
        ax.axvline(0, color="red",
                   linewidth=1.0, linestyle=":",
                   label="Neutral=0")
        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.set_xlabel("Compound Score")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day06_sentiment_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 3: Sentiment vs volatility overlay ─────────────────────
def plot_sentiment_vs_volatility() -> None:
    fig, axes = plt.subplots(len(TICKERS), 1,
                              figsize=(13, 4 * len(TICKERS)))
    fig.suptitle("7-Day Rolling Sentiment vs 21-Day Rolling Volatility",
                 fontsize=13, fontweight="bold")

    for ax, ticker in zip(axes, TICKERS):
        df = load_merged(ticker)
        if df is None or "rv_rolling_21d" not in df.columns:
            continue

        color = TICKER_COL[ticker]
        ax2   = ax.twinx()   # second y-axis for volatility

        # Sentiment on primary axis
        ax.plot(df.index, df["sent_roll7"],
                color=color, linewidth=1.3,
                label="Sentiment (7d MA)", alpha=0.9)
        ax.axhline(0, color="black", linewidth=0.6,
                   linestyle="--", alpha=0.4)
        ax.set_ylabel("Compound Sentiment", color=color)
        ax.tick_params(axis="y", labelcolor=color)

        # Volatility on secondary axis
        ax2.plot(df.index, df["rv_rolling_21d"] * 100,
                 color="grey", linewidth=1.0,
                 linestyle="--", alpha=0.7,
                 label="Vol 21d (%, right)")
        ax2.set_ylabel("Rolling Vol (%)", color="grey")
        ax2.tick_params(axis="y", labelcolor="grey")

        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines[["top"]].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  frameon=False, fontsize=8, loc="upper left")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day06_sentiment_vs_vol.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 4: Correlation heatmap ─────────────────────────────────
def plot_correlation_heatmap() -> None:
    sent_cols = [
        "sent_compound_mean", "sent_roll3", "sent_roll7",
        "sent_pos_mean", "sent_neg_mean", "n_articles",
    ]
    rv_cols = ["rv_rolling_5d", "rv_rolling_21d", "log_return"]

    fig, axes = plt.subplots(1, len(TICKERS),
                              figsize=(5 * len(TICKERS), 5))
    fig.suptitle("Correlation: Sentiment Features vs Volatility Metrics",
                 fontsize=13, fontweight="bold")

    for ax, ticker in zip(axes, TICKERS):
        df = load_merged(ticker)
        if df is None:
            continue

        cols   = [c for c in sent_cols + rv_cols if c in df.columns]
        corr   = df[cols].corr()

        # Show only sentiment vs RV cross-correlations
        s_cols = [c for c in sent_cols if c in corr.columns]
        r_cols = [c for c in rv_cols   if c in corr.index]
        sub    = corr.loc[r_cols, s_cols]

        sns.heatmap(
            sub, ax=ax, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0,
            vmin=-0.5, vmax=0.5,
            linewidths=0.5,
            cbar_kws={"shrink": 0.7},
        )
        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(),
                            rotation=40, ha="right", fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day06_correlation_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 5: Article coverage ────────────────────────────────────
def plot_article_coverage() -> None:
    fig, axes = plt.subplots(len(TICKERS), 1,
                              figsize=(13, 2.5 * len(TICKERS)),
                              sharex=False)
    fig.suptitle("Daily Article Count per Ticker",
                 fontsize=13, fontweight="bold")

    for ax, ticker in zip(axes, TICKERS):
        df = load_merged(ticker)
        if df is None or "n_articles" not in df.columns:
            continue
        color = TICKER_COL[ticker]
        ax.bar(df.index, df["n_articles"],
               color=color, alpha=0.7, width=1.2)
        ax.set_title(ticker, fontsize=10, fontweight="bold")
        ax.set_ylabel("Articles / day")
        ax.grid(True, alpha=0.2, linestyle="--", axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day06_article_coverage.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Main ────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*55}")
    print("  DAY 6 — Generating Plots")
    print(f"{'='*55}")

    plot_sentiment_timeseries()
    plot_sentiment_distribution()
    plot_sentiment_vs_volatility()
    plot_correlation_heatmap()
    plot_article_coverage()

    print("\n  All Day 6 plots complete.")


if __name__ == "__main__":
    main()