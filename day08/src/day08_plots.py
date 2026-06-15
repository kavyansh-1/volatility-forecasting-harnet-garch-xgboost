import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PALETTE = {
    "HAR"    : "#1f77b4",
    "GARCH"  : "#ff7f0e",
    "XGBoost": "#2ca02c",
    "HARNet" : "#9467bd",
}

def _model_color(model_key: str) -> str:
    for k, v in MODEL_PALETTE.items():
        if k.lower() in model_key.lower():
            return v
    return "#888888"

def plot_final_ranking(ranking_csv: str) -> None:
    df = pd.read_csv(ranking_csv)
    df = df.sort_values("composite_score", ascending=True)

    colors = [_model_color(m) for m in df["Model"]]

    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.55)))
    bars = ax.barh(
        range(len(df)), df["composite_score"],
        color=colors, alpha=0.85, edgecolor="white"
    )

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(
            df["composite_score"].max() * 0.02,
            i,
            f"  RMSE={row['RMSE']:.5f}  DirAcc={row['DirAcc']:.1f}%",
            va="center", fontsize=8, color="white", fontweight="bold"
        )

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(
        [f"#{int(r['final_rank'])}  {r['Model']}"
         for _, r in df.iterrows()],
        fontsize=9
    )
    ax.set_xlabel("Composite Rank Score (lower = better)", fontsize=10)
    ax.set_title("Final Model Ranking — All Models",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2, linestyle="--", axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

    legend = [mpatches.Patch(color=v, label=k)
               for k, v in MODEL_PALETTE.items()]
    ax.legend(handles=legend, frameon=False,
               fontsize=8, loc="lower right")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day08_final_ranking.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_dm_heatmap(dm_csv: str) -> None:
    df      = pd.read_csv(dm_csv)
    tickers = df["ticker"].unique()
    n       = len(tickers)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    fig.suptitle("Diebold-Mariano Test p-values\n"
                 "(red < 0.05 = statistically significant difference)",
                 fontsize=12, fontweight="bold")

    for ax, ticker in zip(axes, tickers):
        sub    = df[df["ticker"] == ticker]
        models = sorted(set(sub["label1"].tolist() +
                            sub["label2"].tolist()))

        matrix = pd.DataFrame(np.nan, index=models, columns=models)
        for _, row in sub.iterrows():
            matrix.loc[row["label1"], row["label2"]] = row["p_value"]
            matrix.loc[row["label2"], row["label1"]] = row["p_value"]
        np.fill_diagonal(matrix.values, 1.0)

        sns.heatmap(
            matrix.astype(float),
            ax=ax, annot=True, fmt=".2f",
            cmap="RdYlGn_r",
            vmin=0, vmax=0.2,
            linewidths=0.5,
            cbar=True,
        )
        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(),
                            rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(),
                            rotation=0, fontsize=7)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day08_dm_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")

def plot_regime_rmse(regime_csv: str) -> None:
    df      = pd.read_csv(regime_csv)
    tickers = df["Ticker"].unique()
    models  = df["Model"].unique()
    regimes = ["low", "medium", "high"]
    reg_colors = {"low": "#2ca02c", "medium": "#ff7f0e", "high": "#d62728"}
    n       = len(tickers)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    fig.suptitle("RMSE by Volatility Regime",
                 fontsize=13, fontweight="bold")

    for ax, ticker in zip(axes, tickers):
        sub   = df[df["Ticker"] == ticker]
        x     = np.arange(len(models))
        w     = 0.25

        for i, regime in enumerate(regimes):
            reg_sub = sub[sub["Regime"] == regime]
            vals = []
            for m in models:
                m_sub = reg_sub[reg_sub["Model"] == m]
                vals.append(float(m_sub["RMSE"].iloc[0]) * 1e4
                            if len(m_sub) else np.nan)
            ax.bar(x + i * w, vals, width=w,
                   label=regime, color=reg_colors[regime],
                   alpha=0.80, edgecolor="white")

        ax.set_xticks(x + w)
        ax.set_xticklabels(
            [m.replace("_", "\n") for m in models],
            fontsize=7, rotation=0
        )
        ax.set_ylabel("RMSE × 10⁴")
        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.legend(title="Regime", frameon=False, fontsize=8)
        ax.grid(True, alpha=0.2, linestyle="--", axis="y")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day08_regime_rmse.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")

def plot_rmse_heatmap(ranking_csv: str) -> None:
    df = pd.read_csv(
        ranking_csv.replace("final_ranking", "per_ticker_ranks")
    )
    if "RMSE" not in df.columns or "Ticker" not in df.columns:
        return

    pivot = df.pivot_table(
        index="Model", columns="Ticker",
        values="RMSE", aggfunc="mean"
    ) * 1e4

    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(7, max(4, len(pivot) * 0.5)))
    sns.heatmap(
        pivot, ax=ax,
        annot=True, fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "RMSE × 10⁴"},
    )
    ax.set_title("RMSE × 10⁴ by Model and Ticker\n(lower = better)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day08_rmse_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_radar_chart(ranking_csv: str) -> None:
    df = pd.read_csv(ranking_csv).head(4)

    metrics = ["RMSE", "MAE", "QLIKE", "DirAcc"]
    metrics = [m for m in metrics if m in df.columns]
    N       = len(metrics)
    normed = df[metrics].copy()
    for m in ["RMSE", "MAE", "QLIKE"]:
        if m in normed.columns:
            col = normed[m]
            rng = col.max() - col.min()
            if rng > 0:
                normed[m] = 1 - (col - col.min()) / rng
            else:
                normed[m] = 1.0
    for m in ["DirAcc"]:
        if m in normed.columns:
            col = normed[m]
            rng = col.max() - col.min()
            if rng > 0:
                normed[m] = (col - col.min()) / rng
            else:
                normed[m] = 0.5

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   

    fig, ax = plt.subplots(figsize=(7, 7),
                            subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for _, row in df.iterrows():
        model  = row["Model"]
        values = normed.loc[row.name, metrics].tolist()
        values += values[:1]
        color  = _model_color(model)
        ax.plot(angles, values, color=color,
                linewidth=2, label=model)
        ax.fill(angles, values, color=color, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"],
                        fontsize=7, color="grey")
    ax.grid(True, alpha=0.3)
    ax.set_title("Top 4 Models — Normalised Metric Profile\n"
                 "(outer = better on each axis)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
               frameon=False, fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day08_radar_chart.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_pipeline_timeline() -> None:
    days = [
        ("Day 1", "Data Download\nOHLCV → raw CSVs",         "#aec7e8"),
        ("Day 2", "Feature Engineering\nReturns + Vol Est.",  "#ffbb78"),
        ("Day 3", "Diagnostics\nADF + ARCH-LM + ACF",        "#98df8a"),
        ("Day 4", "Baseline Models\nHAR + GARCH + XGB",      "#ff9896"),
        ("Day 5", "HARNet CNN\nPyTorch + Training",           "#c5b0d5"),
        ("Day 6", "Sentiment Pipeline\nFinBERT + Merge",      "#c49c94"),
        ("Day 7", "Augmented Models\nSentiment + DM Test",    "#f7b6d2"),
        ("Day 8", "Final Evaluation\nRanking + Regimes",      "#dbdb8d"),
    ]

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.set_xlim(-0.5, len(days) - 0.5)
    ax.set_ylim(0, 2)
    ax.axis("off")
    ax.set_title("Project Pipeline — Days 1 to 8",
                 fontsize=13, fontweight="bold", pad=12)

    for i, (label, desc, color) in enumerate(days):
        rect = mpatches.FancyBboxPatch(
            (i - 0.4, 0.3), 0.8, 1.4,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="grey",
            linewidth=1.2, alpha=0.9,
        )
        ax.add_patch(rect)
        ax.text(i, 1.55, label,
                ha="center", va="center",
                fontsize=9, fontweight="bold")
        ax.text(i, 0.90, desc,
                ha="center", va="center",
                fontsize=7.5, linespacing=1.4)
        if i < len(days) - 1:
            ax.annotate(
                "", xy=(i + 0.42, 1.0), xytext=(i + 0.58, 1.0),
                arrowprops=dict(
                    arrowstyle="->", color="grey",
                    lw=1.5, connectionstyle="arc3,rad=0"
                )
            )

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day08_pipeline_timeline.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")

def run_all_plots(dm_csv:      str = None,
                   ranking_csv: str = None,
                   regime_csv:  str = None):
    print(f"\n{'='*55}")
    print("  DAY 8 — Generating Plots")
    print(f"{'='*55}")

    ranking_csv = ranking_csv or os.path.join(
        OUT_DIR, "day08_final_ranking.csv"
    )
    dm_csv      = dm_csv or os.path.join(
        OUT_DIR, "day08_dm_results.csv"
    )
    regime_csv  = regime_csv or os.path.join(
        OUT_DIR, "day08_regime_analysis.csv"
    )

    if os.path.exists(ranking_csv):
        plot_final_ranking(ranking_csv)
        plot_rmse_heatmap(ranking_csv)
        plot_radar_chart(ranking_csv)

    if os.path.exists(dm_csv):
        plot_dm_heatmap(dm_csv)

    if os.path.exists(regime_csv):
        plot_regime_rmse(regime_csv)

    plot_pipeline_timeline()

    print("\n  All Day 8 plots complete.")