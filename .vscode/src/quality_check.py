import os
import pandas as pd
import numpy as np


def run_quality_check(dfs: dict) -> pd.DataFrame:
    rows = []

    for ticker, df in dfs.items():
        n_rows = len(df)
        start = df.index.min()
        end = df.index.max()

        missing_total = int(df.isnull().sum().sum())
        missing_col = df.isnull().sum().to_dict()

        n_duplicates = int(df.index.duplicated().sum())

        non_bdays = int((~df.index.day_of_week.isin(range(5))).sum())

        negative_prices = int((df[["Open", "High", "Low", "Close"]] < 0).any(axis=1).sum())

        zero_volume = int((df["Volume"] == 0).sum())

        bad_ohlc = int((
            (df["High"] < df["Open"]) |
            (df["High"] < df["Close"]) |
            (df["Low"] > df["Open"]) |
            (df["Low"] > df["Close"])
        ).sum())

        issues = missing_total + n_duplicates + negative_prices + zero_volume + bad_ohlc
        status = "CLEAN" if issues == 0 else f"ISSUES: {issues}"

        rows.append({
            "Ticker": ticker,
            "Status": status,
            "Rows": n_rows,
            "Start": start,
            "End": end,
            "Missing cells": missing_total,
            "Duplicate_dates": n_duplicates,
            "Non_bdays": non_bdays,
            "Neg_Prices": negative_prices,
            "Zero_volume": zero_volume,
            "Bad_OHLC": bad_ohlc,
            "Close_min": round(float(df["Close"].min()), 2) if not df.empty else np.nan,
            "Close_max": round(float(df["Close"].max()), 2) if not df.empty else np.nan,
        })

    report = pd.DataFrame(rows).set_index("Ticker")
    return report


if __name__ == "__main__":
    tickers = ["SPY", "QQQ", "AAPL"]

    # for loading all the csvs
    dfs = {}
    for t in tickers:
        path = f"data/raw/{t}_daily.csv"
        if os.path.exists(path):
            dfs[t] = pd.read_csv(path, index_col="Date", parse_dates=True)
        else:
            print(f"File not found {path} - run download_data.py first")

    report = run_quality_check(dfs)

    print("\n" + "-" * 65)
    print("DATA QUALITY REPORT")
    print("-" * 65)
    print(report.to_string())
    print("-" * 65)

    os.makedirs("reports", exist_ok=True)
    report.to_csv("reports/day-1_quality_report.csv")
    print("\nSaved -> reports/day-1_quality_report.csv successfully")
    





    


