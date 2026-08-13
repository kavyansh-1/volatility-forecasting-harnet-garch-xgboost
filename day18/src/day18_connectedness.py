import os 
import warnings 
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd 
from statsmodels.tsa.api import VAR 

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS = ["SPY", "QQQ", "AAPL"]
FEVD_HORIZON = 10
ROLL_WIN = 252
ROLL_STEP = 21 

def fevd_from_var(rv_window: pd.DataFrame , n_lags : int , horizon : int = FEVD_HORIZON)-> np.ndarray:
    n = len(rv_window)
    k = len(TICKERS)

    if n < (n_lags+1)*k+10:
        return np.eye(k) / k

    try: 
        model = VAR(rv_window)
        result = model.fit(maxlags=n_lags , ic = None , trend = "c")
        fevd = result.fevd(periods = horizon)
        ## decomp has shape (k, periods, k) in current statsmodels; take the last horizon
        return fevd.decomp[:,-1,:] 
    except Exception: 
        return np.eye(k)/k



def diebold_yilmaz_table(fevd_mat: np.ndarray , tickers : list = TICKERS)-> pd.DataFrame:
    n = len(tickers)
    mat = fevd_mat.copy()

    #Normalising rows to sum to 1 (handle numerical precision issues)
    row_sums = mat.sum(axis = 1, keepdims = True)
    mat = mat / np.maximum(row_sums , 1e-10)

    table = pd.DataFrame(mat , index = tickers , columns = tickers)

    #TO : how much asset j is contributing to the others forecast errors 
    # column sum of off diagonal entries 
    to_row = mat.sum(axis=0) - np.diag(mat)
    table.loc["TO"] = to_row

    #FROM: how much asset i receives from others 
    # this is the row sum of off diagonal entries 
    from_col = mat.sum(axis=1)-np.diag(mat)
    table["FROM"] = np.append(from_col , np.nan)
    table.loc["TO","FROM"] = np.nan

    #NET
    net = to_row - from_col
    table.loc["NET"] = np.append(net , np.nan)

    #Now the total correctedness Index
    tci = (mat.sum() - np.trace(mat)) / n
    table.loc["TCI"] = np.nan
    table.loc["TCI" , "FROM"] = tci 

    return table.round(4)

def total_connectedness_index(fevd_mat : np.ndarray)-> float:

    n = fevd_mat.shape[0]
    off_diag = fevd_mat.sum() - np.trace(fevd_mat)
    return float ( off_diag / n)

def rolling_connectedness(rv_df : pd.DataFrame , win: int = ROLL_WIN , step : int = ROLL_STEP, n_lags : int = 1 , horizon : int = FEVD_HORIZON)-> pd.DataFrame:
    n = len(rv_df)
    rows = []

    for end in range(win , n , step): 
        start = end - win 
        window = rv_df.iloc[start:end]
        date = rv_df.index[end-1]

        fevd_mat = fevd_from_var(window , n_lags = n_lags , horizon = horizon)
        tci = total_connectedness_index(fevd_mat)

        # Normalise 
        row_sums = fevd_mat.sum(axis = 1 , keepdims = True)
        mat_norm = fevd_mat / np.maximum(row_sums , 1e-10)
        to = mat_norm.sum(axis = 0) - np.diag(mat_norm)
        from_ = mat_norm.sum(axis = 1) - np.diag(mat_norm)
        net = to - from_

        row = {"date" : date, "TCI" : round(tci , 4)}
        for i , t in enumerate(TICKERS):
            row[f"TO_{t}"] = round(float(to[i]) , 4)
            row[f"FROM_{t}"] = round(float(from_[i]), 4)
            row[f"NET_{t}"]  = round(float(net[i]),   4) 
        rows.append(row) 

    return pd.DataFrame(rows).set_index("date")

def run_connectedness(var_results : dict)-> dict: 
    print(f"\n{'='*55}")
    print("  DAY 18 — Diebold-Yilmaz Connectedness")
    print(f"{'='*55}")

    rv_df = var_results["rv_df"]
    fevd_mat = var_results["fevd_df"].values[:len(TICKERS), : len(TICKERS)]

 # Full-sample table
    dy_table = diebold_yilmaz_table(fevd_mat)
    tci_full = total_connectedness_index(fevd_mat)

    print(f"\n  Diebold-Yilmaz Table (FEVD at {FEVD_HORIZON}-step horizon):")
    print(dy_table.to_string())
    print(f"\n  Total Connectedness Index (TCI) = {tci_full:.4f}")
    print(f"  Interpretation: {tci_full*100:.1f}% of each asset's forecast"
          f" error variance\n  is explained by shocks in the other assets.")

    # Rolling connectedness
    print(f"\n  Computing rolling TCI ({ROLL_WIN}-day window, "
          f"{ROLL_STEP}-day step)...")
    roll_df = rolling_connectedness(rv_df)
    print(f"  TCI range: [{roll_df['TCI'].min():.4f}, "
          f"{roll_df['TCI'].max():.4f}]")
    print(f"  TCI mean : {roll_df['TCI'].mean():.4f}")

    # Save
    dy_table.to_csv(os.path.join(OUT_DIR, "day18_dy_table.csv"))
    roll_df.to_csv( os.path.join(OUT_DIR, "day18_rolling_tci.csv"))

    print(f"\n  [OK] day18_dy_table.csv")
    print(f"  [OK] day18_rolling_tci.csv")

    return {
        "dy_table": dy_table,
        "tci_full": tci_full,
        "roll_df" : roll_df, }  
