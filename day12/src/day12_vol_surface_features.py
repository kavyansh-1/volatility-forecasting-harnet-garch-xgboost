import os 
import numpy as np 
import pandas as pd 

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR , ".." , "data" , "processed")
OUT_DIR = os.path.join(BASE_DIR , "output")

TICKERS = ["SPY" , "QQQ" , "AAPL"]

def add_vol_surface_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rv_1d = df["log_return"] ** 2 * 252
    if "rv_rolling_5d" in df.columns and "rv_rolling_21d" in df.columns:
        df["ts_slope_5_21"] = (df["rv_rolling_5d"].shift(1) / (df["rv_rolling_21d"].shift(1) + 1e-10))
    
    if "rv_rolling_21d" in df.columns and "rv_rolling_63d" in df.columns:
        df["ts_slope_21_63"] = (
            df["rv_rolling_21d"].shift(1) / (df["rv_rolling_63d"].shift(1) + 1e-10)
        )
    
    if "rv_rolling_5d" in df.columns and "rv_rolling_63d" in df.columns:
        df["ts_slope_5_63"] = (
            df["rv_rolling_5d"].shift(1) / (df["rv_rolling_63d"].shift(1) + 1e-10)
        )

    for w in [10,21]:
        df[f"vov_{w}d"] = (
            rv_1d.shift(1).rolling( w , min_periods = w).std()
        )
    
    if "park_vol_21d" in df.columns and "gk_vol_21d" in df.columns:
        df["park_gk_spread_21d"] = (df["gk_vol_21d"].shift(1) - df["park_vol_21d"].shift(1))
    
    df["park_gk_spread_norm"] = (df["park_gk_spread_21d"] / (df["gk_vol_21d"].shift(1) + 1e-10))
    

    df["realized_skew_21d"] = (df["log_return"].shift(1).rolling(21 , min_periods = 15).skew())
    
    df["realized_kurt_21d"] = df["log_return"].shift(1).rolling(21, min_periods=15).kurt()

    for w in [21]:
        r_lag = df["log_return"].shift(1)
        up_rv = (r_lag.clip(lower = 0) ** 2 * 252).rolling(w , min_periods = 10).mean()
        dn_rv = (r_lag.clip(upper=0) ** 2 * 252).rolling(w, min_periods=10).mean()
        df[f"updown_vol_ratio_{w}d"] = up_rv / (dn_rv + 1e-10)

    roll_std = df["log_return"].shift(1).rolling(21, min_periods=10).std()
    jump_flag = (df["log_return"].shift(1).abs() > 3 * roll_std).astype(float)
    df["jump_count_21d"] = jump_flag.rolling(21 , min_periods = 10).sum()
    
    return df 

def run_vol_surface_features() -> dict:
    print(f"\n{'='*55}")
    print("  DAY 12 — Volatility Surface Features")
    print(f"{'='*55}")

    results = {}
    for ticker in TICKERS:
        path = os.path.join(DATA_DIR , f"{ticker}_processed.csv")
        if not os.path.exists(path):
            print(f" {path} not found , run day 2 first ! ")
            continue 
        df = pd.read_csv(path , index_col = "Date" , parse_dates = True)
        df = add_vol_surface_features(df)

        new_cols = [
            "ts_slope_5_21" , "ts_slope_21_63" , "ts_slope_5_63" , "vov_10d" , "vov_21d" , "park_gk_spread_21d", "park_gk_spread_norm", "realized_skew_21d", "realized_kurt_21d", "updown_vol_ratio_21d", "jump_count_21d"
        ]
        new_cols = [c for c in new_cols if c in df.columns]

        print(f"\n{ticker} : {len(new_cols)} vol surface features have been added !")
        print(f" Non null counts (tail check): ")
        print(df[new_cols].tail(5).notnull().sum().to_string())

        results[ticker] = df

    return results 






    