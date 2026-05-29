import numpy as np
import pandas as pd
import os
from pathlib import Path

def compute_returns(df: pd.DataFrame, price_col: str = "Close")-> pd.DataFrame:
    df = df.copy()

    df["log_return"] = np.log(df[price_col]/df[price_col].shift(1))
    df["simple_return"] = df[price_col].pct_change()
    df = df.dropna(subset=["log_return"])

    return df

def compute_all_returns(dfs:dict, price_col : str = "Close") -> dict:

    return {ticker: compute_returns(df,price_col) for ticker , df in dfs.items()}

def describe_returns(dfs:dict)-> pd.DataFrame:
    rows = []
    for ticker , df in dfs.items():
        r = df["log_return"]
        rows.append({
            "Ticker"  : ticker,
            "N"  : len(r),
            "Mean" : round(r.mean() , 6),
            "Std": round(r.std() , 6),
            "Skew": round(r.skew() , 4),
            "Kurt": round(r.kurt() , 4),      ## Excess Kurtosis (Fisher Definition)
            "Min" : round(r.min(),    4),
            "Max" : round(r.max(),    4),
            "Ann_Vol" : round(r.std() * np.sqrt(252), 4),  # Annualised

        })

    return pd.DataFrame(rows).set_index("Ticker")
    
if __name__ == "__main__":
    tickers = ["SPY", "QQQ" , "AAPL"]

    repo_root = Path(__file__).resolve().parents[2]
    dfs = {
        t: pd.read_csv(repo_root / "data" / "raw" / f"{t}_daily.csv", index_col="Date", parse_dates=True)
        for t in tickers
    }

    dfs_ret = compute_all_returns(dfs)

    print("\n Return Statistics")
    print("-"*65)
    print(describe_returns(dfs_ret).to_string())
    print("-"*65)

    ## For my Quick Sanity Check - I wrote this for printing the first few rows
    print("\n SPY sample (first 5 rows with returns are as follows):")
    print(dfs_ret["SPY"][["Close", "log_return", "simple_return"]].head())

