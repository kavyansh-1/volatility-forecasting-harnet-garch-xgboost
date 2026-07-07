import os 
import warnings 
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd
from scipy import stats

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS   = ["SPY", "QQQ", "AAPL"]
TEST_SIZE = 500

def merge_estimators(ticker:str , rv_results: dict , jump_results : str)->pd.DataFrame:
    ## Loading Day 2 Data
    daily_path = os.path.join(DATA_DIR , f"{ticker}_processed.csv")
    daily_df = pd.read_csv(daily_path , index_col ="Date" , parse_dates = True)

    ## Intraday RV estimates now 
    rv_df = rv_results[ticker]
    jump_df = jump_results[ticker]["combined_df"]

    ## Merging on Date Index 

    merged = daily_df.join(rv_df , how="inner")
    merged = merged.join(jump_df[["jump_var" , "jump_ratio" , "jump_detected" , "J_stat" , "p_value"]],
                         how = "left")
    return merged

def qlike_loss(y: np.ndarray , yhat:np.ndarray , floor: float = 1e-8)-> float:
    """QLIKE = mean(log(h) + y/h). Used throughout Days 4-15."""
    h = np.maximum(yhat, floor)
    v = np.maximum(y , floor)

    return float(np.mean(np.log(h) + v/h))

def mse_loss( y: np.ndarray , yhat : np.ndarray)-> float:
    return float(np.mean((y-yhat)**2))

def compare_estimators_as_forecasts(merged: pd.DataFrame , ticker: str)-> pd.DataFrame:
    #Target: next day intraday Rv
    y = merged["rv_5min_ann"].shift(-1).dropna()

    estimators = {}

    #Daily-based estimators (lagged 1)
    for col in ["rv_rolling_21d" , "park_vol_21d" , "gk_vol_21d"]:
        if col in merged.columns:
            estimators[col] = merged[col].shift(1)

    # INTRADAY BASED ESTIMATORS (LAGGED BY 1)
    for col in ["rv_5min_ann" , "bv_5min_ann" , "rk_5min_ann"]:
        if col in merged.columns:
            estimators[col] = merged[col].shift(1)

    # ALIGN ALL OF THESE TO THE TARGET INDEX 
    common = y.index
    rows = []
    for name , forecast in estimators.items():
        fc = forecast.reindex(common).dropna()
        yt = y.reindex(fc.index).dropna()
        if len(yt) < 10:
            continue
        rows.append({
            "TICKER" : ticker, 
            "ESTIMATOR": name, 
            "MSE" : round(mse_loss(yt.values , fc.values), 8),
            "Q-LIKE" : round(qlike_loss(yt.values , fc.values) , 6), 
            "N" : len(yt)

        })
    return pd.DataFrame(rows)

def analyse_jump_day_performance( merged: pd.DataFrame , ticker : str)-> pd.DataFrame:

    target = merged["rv_5min_ann"].shift(-1)
    is_jump = merged["jump_detected"].fillna(0).astype(bool)

    rows = []
    for jump_flag , label in [(False , "No-Jump") , (True , "Jump")]:
        mask = (is_jump==jump_flag) & target.notna()
        if mask.sum() < 5:
            continue
        y = target[mask].values

        for col in ["rv_5min_ann" , "bv_5min_ann" , "gk_vol_21d"]:
            if col not in merged.columns:
                continue 
            fc = merged[col].shift(1)[mask].values
            valid = ~(np.isnan(fc) | np.isnan(y))
            if valid.sum() < 5:
                continue
            rows.append({
                "Ticker"    : ticker,
                "Day_type"  : label,
                "Estimator" : col,
                "QLIKE"     : round(qlike_loss(y[valid], fc[valid]), 6),
                "MSE"       : round(mse_loss(y[valid], fc[valid]), 8),
                "N"         : int(valid.sum()),
            })
    
    return pd.DataFrame(rows)

def run_estimator_comparison(rv_results : dict , jump_results : dict)-> dict:
    """Compare all estimators for all tickers."""
    print(f"\n{'='*55}")
    print("  DAY 16 — Estimator Comparison")
    print(f"{'='*55}")

    all_compare_rows = []
    all_jump_rows = []

    for ticker in TICKERS:
        if ticker not in rv_results or ticker not in jump_results:
            continue

        print(f"\n{ticker}")
        merged = merge_estimators(ticker , rv_results , jump_results)

        compare_df = compare_estimators_as_forecasts(merged, ticker)
        jump_df = analyse_jump_day_performance(merged , ticker)

        all_compare_rows.append(compare_df)
        all_jump_rows.append(jump_df)

        print(f"\n QLIKE comparison (lower = better):")
        print(compare_df[["Estimator" , "QLIKE" , "MSE"]].sort_values("QLIKE").to_string(index = False))

    compare_all = pd.concat(all_compare_rows , ignore_index = True)
    jump_all = pd.concat(all_jump_rows , ignore_index = True)

    compare_all.to_csv(os.path.join(OUT_DIR , "day16_estimator_comparison.csv"), index = False)
    jump_all.to_csv(os.path.join(OUT_DIR , "day16_jump_day_analysis.csv"), index = False)


    print(f"\n  ✓ day16_estimator_comparison.csv")
    print(f"  ✓ day16_jump_day_analysis.csv")

    return {"compare_df": compare_all, "jump_df": jump_all}

