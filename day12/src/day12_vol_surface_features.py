import os 
import numpy as np 
import pandas as pd 

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR , ".." , "data" , "processed")
OUT_DIR = os.path.join(BASE_DIR , "output")

TICKERS = ["SPY" , "QQQ" , "AAPL"]

def add_vol_surface_features ( df:pd.DataFrame)-> pd.DataFrame:
    df = df.copy()
    rv_1d = df["log_return"] ** 2 * 252
    if "rv_rolling_5d" in df.columns and "rv_rolling_21d" in df.columns:
        df["ts_slope_5_21"] = df["rv_rolling_5d"].shift(1) / (df["rv_rolling_21d"].shift(1) + 1e-10)
    
    if "rv_rolling_21d" in df.columns and "rv_rolling_63d" in df.columns:
        df["ts_slope_21_63"] = (
            df["rv_rolling_21d"].shift(1) / (df["rv_rolling_63d"].shift(1) + 1e-10)
        )
    
    if "rv_rolling_5d" in df.columns and "rv_rolling_63d" in df.columns:
        df["ts_slope_5_63"] = (
            df["rv_rolling_5d"].shift(1) / (df["rv_rolling_63d"].shift(1) + 1e-10)
        )

    return df    