import os 
import warnings 
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd 
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS = ["SPY" , "QQQ" , "AAPL"]
MAX_LAGS = 10 
IRF_STEPS = 20 #Impulse response horizon (trading days)
FEVD_HORIZON = 10

def load_rv_matrix() -> pd.DataFrame:
    series = {}
    for ticker in TICKERS:
        path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
        df = pd.read_csv(path , index_col = "Date", parse_dates=True)
        rv = df["log_return"]**2*252
        series[ticker] = rv.rename(ticker)
    return pd.concat(series , axis=1).dropna(how="any")

def check_stationarity(rv_df : pd.DataFrame)->dict:
    results = {}
    for col in rv_df.columns:
        stat , pval, _, _, _, _ = adfuller(rv_df[col].dropna() , maxlag = 5)
        results[col] = {
            "adf_stat" : round(float(stat) , 4), 
            "p_value" : round(float(pval) , 4), 
            "stationary" : bool(pval < 0.05),
        }
        print(f"ADF {col}: stat = {stat:.4f} p = {pval:.4f}"
              f"{'stationary [OK]' if pval < 0.05 else 'non-stationary [FAIL]'}")
    return results 

def select_var_lag(rv_df: pd.DataFrame, max_lags: int = MAX_LAGS)-> int:

    model = VAR(rv_df)
    results = model.select_order(maxlags=max_lags)
    best_lag = int(results.aic)  # lag order selected by AIC (statsmodels 0.14: scalar order)
    best_lag = max(1 , min(best_lag , max_lags))
    return best_lag

def fit_var(rv_df: pd.DataFrame, n_lags: int)-> object:
    model = VAR(rv_df)
    result = model.fit(maxlags=n_lags , ic = None, trend="c")
    return result

def extract_coefficient_matrix(var_result, lag: int = 1) -> pd.DataFrame:
    coefs = var_result.coefs # shape: (n_lags , n_assets , n_assets)
    mat = coefs[lag-1] # lag-1 coeffecient matrix 
    return pd.DataFrame(mat , index = TICKERS , columns=TICKERS)

def compute_irf(var_result, periods:int = IRF_STEPS)-> dict:
    irf = var_result.irf(periods=periods)
    irfs = {}
    for i , ticker_i in enumerate(TICKERS):
        rows = []
        for t in range(periods+1):
            row = {ticker_j: float(irf.orth_irfs[t,i,j]) for j , ticker_j in enumerate(TICKERS)}
            row["period"] = t
            rows.append(row)
        irfs[ticker_i] = pd.DataFrame(rows).set_index("period")

    return irfs

def compute_fevd(var_result , horizon: int = FEVD_HORIZON)->pd.DataFrame:
    fevd_result = var_result.fevd(periods=horizon) # decomp shape: (n_assets, horizon, n_assets)
    # decomp[i, h, j] = fraction of asset i's forecast-error variance at horizon h from shock j

    #Taking the final horizon value (most informative for the long run spillovers)
    decomp = fevd_result.decomp[:, -1, :] #(n_assets , n_assets)

    fevd_df = pd.DataFrame( decomp , index = [f"{t}_receives" for t in TICKERS], 
                                     columns = [f"{t}_sends" for t in TICKERS],
    )
    return fevd_df


def run_var_model() -> dict:
    """Full VAR pipeline: load → stationarity → select lag → fit → IRF → FEVD."""
    print(f"\n{'='*55}")
    print("  DAY 18 — VAR Model for Volatility Dynamics")
    print(f"{'='*55}")

    rv_df = load_rv_matrix()
    print(f"\n  RV panel: {len(rv_df)} days × {len(TICKERS)} assets")

    # Stationarity checks
    print("\n  ADF stationarity tests:")
    adf_results = check_stationarity(rv_df)

    # Lag selection
    print("\n  Selecting VAR lag order by AIC...")
    n_lags = select_var_lag(rv_df)
    print(f"  AIC-optimal lag: {n_lags}")

    # Fit VAR
    print(f"\n  Fitting VAR({n_lags})...")
    var_result = fit_var(rv_df, n_lags)
    print(f"  Log-likelihood : {var_result.llf:.2f}")
    print(f"  AIC            : {var_result.aic:.2f}")

    # Lag-1 coefficient matrix
    coef_mat = extract_coefficient_matrix(var_result, lag=1)
    print(f"\n  VAR(1) coefficient matrix:")
    print(coef_mat.round(4).to_string())

    # IRF
    print(f"\n  Computing IRFs ({IRF_STEPS} periods)...")
    irfs = compute_irf(var_result)

    # FEVD
    print(f"  Computing FEVD ({FEVD_HORIZON}-step horizon)...")
    fevd_df = compute_fevd(var_result)
    print(f"\n  FEVD at horizon {FEVD_HORIZON}:")
    print(fevd_df.round(4).to_string())

    # Save
    coef_mat.to_csv(os.path.join(OUT_DIR, "day18_var_coefficients.csv"))
    fevd_df.to_csv(os.path.join(OUT_DIR, "day18_fevd.csv"))

    # Save IRFs
    irf_rows = []
    for resp_ticker, irf_df in irfs.items():
        for shock_ticker in TICKERS:
            for period, val in irf_df[shock_ticker].items():
                irf_rows.append({
                    "response": resp_ticker,
                    "shock"   : shock_ticker,
                    "period"  : period,
                    "irf"     : val,
                })
    irf_out = pd.DataFrame(irf_rows)
    irf_out.to_csv(os.path.join(OUT_DIR, "day18_irf.csv"), index=False)

    print(f"\n  [OK] day18_var_coefficients.csv")
    print(f"  [OK] day18_fevd.csv")
    print(f"  [OK] day18_irf.csv")

    return {
        "rv_df"      : rv_df,
        "var_result" : var_result,
        "n_lags"     : n_lags,
        "coef_mat"   : coef_mat,
        "irfs"       : irfs,
        "fevd_df"    : fevd_df,
        "adf"        : adf_results,
    }
    
















