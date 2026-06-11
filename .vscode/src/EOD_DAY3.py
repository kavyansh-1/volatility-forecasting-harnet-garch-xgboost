import sys
import os
import pandas as pd
from pathlib import Path

# Ensure the local `src` package (in .vscode/src) is importable when running
sys.path.insert(0, os.path.dirname(__file__))

from src.stat_tests import (
    adf_test,
    ljung_box_test,
    compute_acf_pacf,
    plot_acf_pacf,
)

DATA_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
PLOTS_DIR = Path("plots/day3")
REPORTS_DIR.mkdir(parents = True, exist_ok = True)
PLOTS_DIR.mkdir(parents = True , exist_ok = True)

TICKERS = ["SPY" , "QQQ" , "AAPL"]

def load_ticker_df(ticker:str)-> pd.DataFrame:
    # processed files are saved as {ticker}_processed.csv
    path = DATA_DIR  / f"{ticker}_processed.csv"
    df = pd.read_csv(path, parse_dates = ["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df 

def diagnostics_for_ticker(ticker:str)-> dict:
    df = load_ticker_df(ticker)
    # processed data uses 'log_return' column
    r_t = df["log_return"]
    r2_t = r_t ** 2

    adf_r = adf_test(r_t , max_lag = 10 , regression = "c")
    adf_r2 = adf_test(r2_t , max_lag = 10, regression = "c")

    lb_r = ljung_box_test(r_t, lags = (5,10,20))
    lb_r2 = ljung_box_test(r2_t , lags = (5,10,20))

    acfpacf_r = compute_acf_pacf(r_t , nlags = 40)
    acfpacf_r2 = compute_acf_pacf(r2_t , nlags = 40)

    plot_acf_pacf(r_t, title_prefix=f"{ticker} returns" , outdir = PLOTS_DIR , nlags = 40)
    plot_acf_pacf(r2_t, title_prefix=f"{ticker} squared returns" , outdir = PLOTS_DIR , nlags = 40)

    return { 
        "ticker": ticker,
        "adf_r": adf_r,
        "adf_r2": adf_r2,
        "lb_r": lb_r,
        "lb_r2": lb_r2,
        "acf_r" : acfpacf_r["acf"],
        "pacf_r": acfpacf_r["pacf"],
        "acf_r2" : acfpacf_r2["acf"],
        "pacf_r2": acfpacf_r2["pacf"],


    } 

def summarise_adf_results(results:list)-> pd.DataFrame:
    rows = []
    for res in results:
        ticker = res["ticker"]
        for series_name, adf_res in [("r_t", res["adf_r"]), ("r_t^2", res["adf_r2"])]:
            rows.append({
                "ticker": ticker,
                "series": series_name,
                "statistic": adf_res["statistic"],
                "pvalue": adf_res["pvalue"],
                "used": adf_res.get("used", adf_res.get("usedlag", None)),
                "nobs": adf_res["nobs"],
                "crit_1pct": adf_res["crit_values"]["1%"],
                "crit_5pct": adf_res["crit_values"]["5%"],
                "crit_10pct": adf_res["crit_values"]["10%"],
                "icbest": adf_res["icbest"],
            })

    return pd.DataFrame(rows)
        
def summarise_ljungbx_results(results:list)-> pd.DataFrame:
    rows = []
    for res in results:
        ticker = res["ticker"]
        for series_name, lb_df in [("r_t", res["lb_r"]), ("r_t^2", res["lb_r2"])]:
            for lag, row in lb_df.iterrows():
                rows.append({
                    "ticker": ticker,
                    "series": series_name,
                    "lag": lag,
                    "lb_stat": row["lb_stat"],
                    "pvalue": row["lb_pvalue"],
                })
    return pd.DataFrame(rows)


def main():
    all_results = []
    for ticker in TICKERS:
        print(f"Running the Day 3 Diagnostics for {ticker}...")
        res = diagnostics_for_ticker(ticker)
        all_results.append(res)

    adf_summary = summarise_adf_results(all_results)
    lb_summary = summarise_ljungbx_results(all_results)

    adf_path = REPORTS_DIR / "day3_adf_results.csv"
    lb_path= REPORTS_DIR / "day3_lb-results.csv"

    adf_summary.to_csv(adf_path , index = False)
    lb_summary.to_csv(lb_path , index = False)

    print(f"ADF results saved to {adf_path}")
    print(f"Ljung-Box results saved to {lb_path}")
    print(f"ACF-PACF plots saved in {PLOTS_DIR}")











