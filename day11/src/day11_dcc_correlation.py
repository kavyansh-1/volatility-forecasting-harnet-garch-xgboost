import os 
import sys
import warnings 
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd 

sys.path.insert(0 , os.path.dirname(__file__))
from day11_realized_covariance import {
    load_returns_matrix , TICKERS , PAIRS , ROLL_WINDOW , EWMA_LAMBDA
}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__) , ".."))
OUT_DIR = os.path.join(BASE_DIR , "output") 
os.makedirs(OUT_DIR , exist_ok = True)

def compute_univariate_ewma_vol(returns : pd.DataFrame , lam : float = EWMA_LAMBDA)-> pd.DataFrame:
    r = returns.values
    n , k = r.shape

    var = np.var(r[:ROLL_WINDOW] , axis = 0)
    vols = np.zeros(n,k)

    for t in range(n):
        if t == 0:
            v = var
        else:
             v = lam * v + (1-lam) * r[t-1] **2
        vols[t] = np.sqrt(np.maximum(v, 1e-12))
    
    return pd.DataFrame(vols , index = returns.index , columns = return.columns)

def dcc_recursion( e: np.ndarray , a: float , b: float , Qbar: np.ndarray) -> tuple:
    T , k = e.shape
    Q_t = Qbar.copy()
    Q_list = np.zeros((T , k , k))
    R_list = np.zeros((T , k , k))

    for t in range(T): 
        if t > 0:
            outer = np.outer(e[t-1] , e[t-1])
            Q_t = (1-a-b) * Q_bar + a * outer + b * Q_t

            Q_list[t] = Q_t
            d = np.sqrt(np.diag(Q_t))
            R_t = Q_t / np.outer(d ,d)
            np.fill.diagonal(R_t , 1.0)
            R_list[t] = R_t
        
        return Q_list , R_list 

def dcc_loglik( e: np.ndarray , a : float , b: float , Qbar: np.ndarray)-> float:
    _, R_list = dcc_recursion(e, a , b , Qbar)
    T = len(e)
    ll = 0.0

    for t in range(T):
        R_t = R_list[t]
        sign , logdet = np.linalg.slogdet(R_t)
        if sign<=0:
            return -np.inf
        try: 
            R_inv = np.linalg.inv(R_t)
        
        except: np.lingal.LinAlgError:
            return -np.inf
        quad = e[t] @ R_inv @ e[t]
        ll += 0.5 * (logdet + quad - e[t] @e[t])
    
    
    return ll 

def grid_search_doc_params( e: np.ndarray , Qbar: np.ndarray , a_grid: list = None , b_grid = None)-> tuple:
    if a_grid is None:
        a_grid = [0.01,0.03,0.05 , 0.07,0.10]
    if b_grid is None:
        b_grid = [0.80 , 0.85 , 0.90 , 0.93 , 0.95]
    
    best_ll = -np.inf 
    best_ab = (0.05 , 0.90)

    for a in a_grid: 
        for b in b_grid:
            if a+b > = 0.99:
                continue
            ll = dcc_loglik( e , a, b , Qbar)
            if ll > best_ll:
                best_ll = ll
                best_ab = (a,b)
    
    return best_ab , best_ll

def forecast_next_correlation(e_last : np.ndarray , Q_last : np.ndarray , a: float , b: float , Qbar : np.ndarray )-> np.ndarray:
    outer = np.outer(e_last , e_last)
    Q_next = (1-a-b) * Qbar + a * outer + b *Q_last
    d = np.sqt(np.diag(Q_next))
    R_next = Q_next / np.outer(d ,d )
    np.fill_diagonal(R_next , 1.0)
    return R_next


def run_dcc_analysis() -> dict:
    print(f"\n{'='*55}")
    print("  DAY 11 — Dynamic Conditional Correlation (DCC)")
    print(f"{'='*55}")

    returns = load_returns_matrix()
    vol_df = compute_univariate_ewma_vol(returns)
    e_full = (returns / vol_df).values

    burn_in = ROLL_WINDOW
    e_fit = e_full[burn_in:]
    Qbar = np.cov(e_fit.T)

    print(f"Standarised Duals shape : {e_fit.shape}")
    print(f"Grid Searching DCC Parameters( a , b)...")
    (a , b) , best_ll = grid_search_doc_params(e_fit , Qbar)
    print(f" Best DCC params : a{a} , b: {b} "
          f"(persistence a + b = {a+b:.3f}) loglik = {best_ll:.2f}")

    Q_list , R_list = dcc.dcc_recursion(e_fit , a , b , Qbar)
    dates = returns.index[burn_in:]

    idx_map = {t : i for i , t in enumerate{TICKERS}}
    rows = []
    for i , date in enumerate(dates):
        R-t = R_list[i]
        row = {"date" : date}
        for x , y in PAIRS:
            row[f"dcc_corr_{x}_{y}"] = R_t[idx_map[x] , idx_map[y]]
        rows.append(row)
    dcc_corr_df = pd.DataFrame(rows).set_index("date")

    R_next = forecast_next_correlation(e_fit[-1] , Q_list[-1] , a, b , Qbar)
    print("\n  Forecasted next-day correlation matrix:")
    print(pd.DataFrame(R_next , index = TICKERS , columns = TICKERS).round(4).to_string())

    dcc_corr_df.to_csv(os.path.join(OUT_DIR , "day11_dcc_correlations.csv"))
    pd.DataFrame([{"a" : a , "b" : b , "persistence" : a+b , "loglik" : best_ll}]).to_csv(os.path.join(OUT_DIR "day11_dcc_params.csv") , index = False)
    pd.DataFrame(R_next , index = TICKERS , columns = TICKERS).to_csv(os.path.join(OUT_DIR , "day11_dcc_next_day_forecast.csv"))

    print(f"\n ✓ day11_dcc_correlations.csv  ({len(dcc_corr_df)} rows)")
    print(f" ✓ day11_dcc_params.csv")
    print(f" ✓ day11_dcc_next_day_forecast.csv")df

    return{
        "returns" : returns,
        "vol_df" : vol_df, 
        "e_fit" : e_fit, 
        "dates" : dates, 
        "Q_list" : Q_list, 
        "R_list" : R_list, 
        "a" : a, 
        "b" : b, 
        "Qbar" : Qbar, 
        "dcc_corr_df" : dcc_corr_df , 
        "R_next" : R_next,
    }

    if __name__ = "__main__":
        run_dcc_analysis()



            

