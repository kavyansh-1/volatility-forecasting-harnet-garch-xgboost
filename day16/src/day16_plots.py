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

TICKERS    = ["SPY", "QQQ", "AAPL"]
TICKER_COL = {"SPY" : "#1f77b4" , "QQQ" : "#ff7f0e" , "AAPL" : "#2ca02c"}
EST_COL = {
    "rv_5min_ann" : "#9467bd", 
    "bv_5min_ann" : "#1f77b4", 
    "rk_5min_ann" : "#17becf", 
    "rv_rolling_21d" : "#7f7f7f", 
    "park_vol_21d" :  "#ff7f0e", 
    "gk_vol_21d" : "#2ca02c", 


}

# Plot 1 :- Intraday U-shaped vol pattern 
def plot_intraday_pattern(intraday_data:dict , ticker : str = "SPY")->None:
    if ticker not in intraday_data: return 
    intra = intraday_data[ticker]

    avg_abs = intra.groupby("bar")["log_ret"].apply(lambda x: np.abs(x).mean())

    fig, ax = plt.subplots(figsize = (10,4))
    ax.bar(avg_abs.index , avg_abs.values * 100 , color = TICKER_COL[ticker], alpha = 0.75 , edgecolor = "none")
    ax.plot(avg_abs.index , avg_abs.values * 100 , color = "black" , linewidth = 1.2 , alpha = 0.8)
    ax.set_xlabel("Bar Index (0 = 9:30 AM , 77 = 3:55 PM)")
    ax.set_ylabel("Mean |5-min return| (%)")
    ax.set_title(f"{ticker} - Intraday Volatility Pattern (U-Shape)" , fontsize = 12 , fontweight = "bold")
    ax.grid(True , alpha = 0.2 , linstyle = "--" , axis = "y")
    ax.spines[["top" , "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"day16_intraday_pattern_{ticker}.png")
    plt.savefig(out , dpi = 150 , bbox_inches = "tight")
    plt.close()
    print(f"Saved → {out}")

# Plot 2 :- RV vs BV vs close to close time series
def plot_rv_bv_comparison(rv_results: dict , ticker : str = "SPY")->None:
    if ticker not in rv_results: return 
    df = rv_results[ticker].dropna(subset=["rv_5min_ann","bv_5min_ann"])

    fig , ax = plt.subplots(figsize = (13, 5))
    ax.plot(df.index , df["rv_5min_ann"] * 100, color = EST_COL["rv_5min_ann"], linewidth = 1.0 , alpha = 0.85 , label = "RV(5-min intraday)")
    ax.plot(df.index , df["bv_5min_ann"] * 100 , color = EST_COL["bv_5min_ann"], linewidth = 1.0 , alpha = 0.85, label = "BV (jump-robust)")

    #Shade jump component: the gap between RV and BV
    ax.fill_between(df.index , df["bv_5min_ann"]*100 , df["rv_5min_ann"]*100 , alpha = 0.25 , color = "red" , label = "Jump Component(RV-BV)")
    ax.set_title(f"{ticker} - Realized Variance vs Bipower Variation", fontsize = 12 , fontweight = "bold")
    ax.set_ylabel("Annualised Vol Estimate (%)")
    ax.legend(frameon = False , fontsize = 9)
    ax.grid(True , alpha = 0.2 , linestyle="--")
    ax.spines[["top" , "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"day16_rv_bv_comparison_{ticker}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")

#Plot - 3 Jump Variance over time 
def plot_jump_variance(jump_results: dict)->None:
    n = len(TICKERS)
    fig , axes = plt.subplots(n , 1 , figsize = (13 , 3.5*n))
    if n == 1:
        axes = [axes]
    fig.suptitle("Daily Jump Variance (RV-BV , positive part)" , fontsize = 13 , fontweight= "bold")

    for ax , ticker in zip(axes , TICKERS):
        if ticker not in jump_results:
            continue 
        df = jump_results[ticker]["combined_df"]
        jv = df["jump_var"].fillna(0)
        jdays = df["jump_detected"].fillna(0).astype(bool)
        color = TICKER_COL[ticker]

        ax.fill_between(df.index , jv * 100 , color = color , alpha = 0.4)
        ax.plot(df.index , jv * 100 , color = color , linewidth = 0.8)

        #Highlight confirmed jump days
        ax.scatter(df.index[jdays] , jv[jdays] * 100 , color = "red" , s = 15 , zorder = 5 , label = "BNS Jump Detected")
        ax.set_title(ticker , fontsize = 11 , fontweight = "bold")
        ax.set_ylabel("Jump var × 100")
        ax.legend(frameon = False , fontsize = 8)
        ax.grid(True , alpha = 0.2 , linestyle = "--")
        ax.spines[["top" , "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day16_jump_variance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")

#Plot 4 The Jump Ratio Distribution
def plot_jump_ratio_dist(jump_results: dict)-> None:
    n = len(TICKERS)
    fig , axes = plt.subplots(1 , n , figsize = (5*n , 4))
    if n == 1: axes = [axes]
    fig.suptitle("Distribution of Jump Ratio (Jump Var / Total RV)" , fontsize = 13 , fontweight = "bold")
    
    for ax , ticker in zip(axes, TICKERS):
        if ticker not in jump_results: continue
        jr = jump_results[ticker]["combined_df"]["jump_ratio"].dropna()
        color = TICKER_COL[ticker]

        ax.hist(jr , bins = 40 , color = color , alpha = 0.65 , edgecolor = "none" , density = True)
        ax.axvline(jr.mean(), color = "black" , linewidth = 1.5, linestyle = "--" , label = f"Mean={jr.mean():.3f}")
        ax.axvline(0.10 , color="red" , linewidth = 1.0, linestyle=":", alpha = 0.6, label = "10% threshold")

        ax.set_title(ticker , fontsize = 11 , fontweight = "bold")
        ax.set_xlabel("Jump Ratio")
        ax.set_ylabel("Density")
        ax.legend(frameon = False , fontsize = 8)
        ax.grid(True , alpha = 0.2 , linestyle = "--")
        ax.spines[["top", "right"]].set_visible(False)
    
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day16_jump_ratio_dist.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")

# Plot 5: Estimator QLIKE bar chart
def plot_estimator_qlike(compare_csv: str) -> None:
    df = pd.read_csv(compare_csv)
    tickers   = sorted(df["Ticker"].unique())
    estimators= df["Estimator"].unique()
    x  = np.arange(len(tickers))
    w  = 0.12

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, est in enumerate(estimators):
        vals = [
            df[(df["Ticker"]==t) & (df["Estimator"]==est)]["QLIKE"].values[0]
            if len(df[(df["Ticker"]==t) & (df["Estimator"]==est)]) > 0
            else np.nan
            for t in tickers
        ]
        color = EST_COL.get(est, "#888888")
        ax.bar(x + i*w, vals, width=w, label=est,
               color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x + w * (len(estimators)-1) / 2)
    ax.set_xticklabels(tickers, fontsize=11)
    ax.set_ylabel("QLIKE Loss (lower = better)")
    ax.set_title("Estimator Comparison: QLIKE vs Next-Day Intraday RV",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    ax.grid(True, alpha=0.2, linestyle="--", axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "day16_estimator_qlike.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# Plot 6: Jump vs no-jump vol distributions
def plot_jump_day_vol_dist(jump_results: dict,
                            ticker: str = "SPY") -> None:
    if ticker not in jump_results: return
    df   = jump_results[ticker]["combined_df"]
    jump = df[df["jump_detected"] == 1]["rv_5min_ann"].dropna()
    nojump = df[df["jump_detected"] != 1]["rv_5min_ann"].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(nojump * 100, bins=40, color="#1f77b4", alpha=0.55,
            density=True, label=f"No-jump days (n={len(nojump)})")
    ax.hist(jump * 100, bins=20, color="#d62728", alpha=0.55,
            density=True, label=f"Jump days (n={len(jump)})")

    ax.axvline(nojump.mean() * 100, color="#1f77b4", linewidth=1.5,
               linestyle="--")
    ax.axvline(jump.mean() * 100, color="#d62728", linewidth=1.5,
               linestyle="--")

    ax.set_title(f"{ticker} — RV Distribution: Jump vs No-Jump Days",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Intraday RV (annualised, %)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"day16_jump_vol_dist_{ticker}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# Main 
def run_all_plots(intraday_data: dict,
                   rv_results:   dict,
                   jump_results: dict) -> None:
    print(f"\n{'='*55}")
    print("  DAY 16 — Generating Plots")
    print(f"{'='*55}")

    for ticker in TICKERS:
        plot_intraday_pattern(intraday_data, ticker=ticker)
        plot_rv_bv_comparison(rv_results, ticker=ticker)
        plot_jump_day_vol_dist(jump_results, ticker=ticker)

    plot_jump_variance(jump_results)
    plot_jump_ratio_dist(jump_results)

    compare_csv = os.path.join(OUT_DIR, "day16_estimator_comparison.csv")
    if os.path.exists(compare_csv):
        plot_estimator_qlike(compare_csv)

    print("\n  All Day 16 plots complete.")


if __name__ == "__main__":
    print("Day 16 plots module loaded. Import the plotting helpers from an orchestrator.")





