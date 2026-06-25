import os 
import warnings
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd 

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__) , ".."))
DATA_DIR = os.path.join(BASE_DIR , ".." , "data" , "processed")
OUT_DIR = os.path.join(BASE_DIR , "output")
os.makedirs(OUT_DIR , exist_ok = True)

TICKERS = ["SPY", "QQQ", "AAPL"]
ROLL_WINDOW = 21 
EWMA_LAMBDA = 0.94
PAIRS = [("SPY", "QQQ"), ("SPY", "AAPL"), ("QQQ", "AAPL")]

def load_returns_matrix() -> pd.DataFrame:
    series = {} 
    for ticker in TICKERS:
        path = os.path.join(DATA_DIR , f"{ticker}_processed.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing Path : {path} - run Day 2 code first please")
        df = pd.read_csv(path , index_col="Date" , parse_dates = True)
        series[ticker] = df["log_return"]
     
    returns = pd.DataFrame(series).dropna(how = "any")
    return returns 

def rolling_covariance(returns : pd.DataFrame , window : int = ROLL_WINDOW)-> dict:
    shifted = returns.shift(1) 
    cov_dict = {}
    
    for i in range(window , len(shifted)):
        window_data = shifted.iloc[i-window : i]
        if window_data.isnull().any().any():
            continue
        cov = window_data.cov().values * 252
        cov_dict[shifted.index[i]] = cov 

    return cov_dict

def ewma_covariance ( returns : pd.DataFrame , lam : float = EWMA_LAMBDA)-> dict:
    r = returns.values
    n , k = r.shape

    sigma = np.cov(r[:ROLL_WINDOW].T) * 252
    ewma_dict = {}
    for t in range(ROLL_WINDOW, n):
        r_prev = r[t-1].reshape(-1, 1)
        sigma = lam * sigma + (1 - lam) * (r_prev @ r_prev.T)
        ewma_dict[returns.index[t]] = sigma * 252

    return ewma_dict

def covariance_to_correlation(cov: np.ndarray)-> np.ndarray:
    d = np.sqrt(np.diag(cov))
    outer = np.outer(d , d)
    corr = cov / np.maximum(outer , 1e-12)
    np.fill_diagonal(corr,1.0)
    return corr 

def extract_pairwise_correlations(cov_dict: dict, pairs: list = PAIRS , tickers : list = TICKERS)-> pd.DataFrame:
    idx_map = {t : i for i , t in enumerate(tickers)}
    rows = []

    for date , cov  in cov_dict.items():
        corr = covariance_to_correlation(cov)
        row = {"date" : date}
        for a , b in pairs: 
            row[f"corr_{a}_{b}"] = corr[idx_map[a] , idx_map[b]]
        rows.append(row)

    return pd.DataFrame(rows).set_index("date").sort_index()


def run_realized_covariance()-> dict:
    print(f"\n{'='*55}")
    print("  DAY 11 — Realized Covariance & Correlation")
    print(f"{'='*55}")

    returns = load_returns_matrix()
    print(f"Aligned returns : {len(returns)} common trading days ")
    print(f"Tickers : {list(returns.columns)}")

    print("\n  Computing rolling covariance (21-day window)...")  
    roll_cov = rolling_covariance(returns , window=ROLL_WINDOW)
    roll_corr_df = extract_pairwise_correlations(roll_cov)

    print("  Computing EWMA covariance (lambda=0.94)...")
    ewma_cov = ewma_covariance( returns , lam = EWMA_LAMBDA)
    ewma_corr_df = extract_pairwise_correlations(ewma_cov)


    roll_corr_df.to_csv(os.path.join(OUT_DIR, "day11_rolling_correlations.csv"))
    ewma_corr_df.to_csv(os.path.join(OUT_DIR, "day11_ewma_correlations.csv"))
    returns.to_csv(os.path.join(OUT_DIR, "day11_aligned_returns.csv"))

    print(f"\n  [OK] day11_rolling_correlations.csv ({len(roll_corr_df)} rows)")
    print(f"\n  [OK] day11_ewma_correlations.csv ({len(ewma_corr_df)} rows)")
    print(f"\n  [OK] day11_aligned_returns.csv ({len(returns)} rows)")

    print("\n Mean Rolling Correlations: ")
    print(roll_corr_df.mean().round(4).to_string())
    print("\n Mean EWMA correlations: ")
    print(ewma_corr_df.mean().round(4).to_string())

    return{
        "returns" : returns,
        "roll_cov" : roll_cov,
        "ewma_cov" : ewma_cov, 
        "roll_corr_df" : roll_corr_df,
        "ewma_corr_df" : ewma_corr_df,
    }

if __name__ == "__main__":
    run_realized_covariance()


