import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

PALETTE = {
    "SPY": "#1F77B4",
    "QQQ": "#FF7F0E",
    "AAPL": "#2CA02C",

}

def plot_adj_close(dfs: dict, save_dir: str = "plots") -> None:
    os.makedirs(save_dir, exist_ok=True)
    tickers = list(dfs.keys())

    fig, axes = plt.subplots(
        nrows=len(tickers),
        ncols=1,
        figsize=(14, 4 * max(1, len(tickers))),
        sharex=True,
    )

    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]

    fig.suptitle(
        "Adjusted Close Prices - SPY - QQQ - AAPL",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    for ax, ticker in zip(axes, tickers):
        df = dfs.get(ticker, pd.DataFrame())
        if df.empty:
            continue
        color = PALETTE.get(ticker, "steelblue")

        ax.plot(df.index, df["Close"], color=color, linewidth=1.2, label=ticker)
        ax.fill_between(df.index, df["Close"], alpha=0.07, color=color)

        ax.set_ylabel("Price (USD)", fontsize=11)
        ax.legend(loc="upper left", fontsize=12, frameon=False)
        ax.grid(True, alpha=0.3, linestyle="--")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Date", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = os.path.join(save_dir, "adj_close_prices.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


def plot_normalised(dfs: dict, save_dir: str = "plots") -> None:
    
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))

    for ticker, df in dfs.items():
        price  = df["Close"]
        normed = price / price.iloc[0] * 100    # Rebase to 100 at start
        ax.plot(normed.index, normed,
                color=PALETTE.get(ticker, "steelblue"),
                linewidth=1.5,
                label=ticker)

    ax.axhline(100, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)  # Base line
    ax.set_title("Normalised Performance — Base 100 (Jan 2015)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Indexed Price", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(fontsize=12, frameon=False)
    ax.grid(True, alpha=0.3, linestyle="--")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()

    out = os.path.join(save_dir, "normalised_performance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


if __name__ == "__main__":
    tickers = ["SPY", "QQQ", "AAPL"]
    dfs = {
        t: pd.read_csv(f"data/raw/{t}_daily.csv", index_col="Date", parse_dates=True)
        for t in tickers
    }
    plot_adj_close(dfs)
    plot_normalised(dfs)
    print("\n Both plots saved to plots/")

    

