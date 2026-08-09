import os 
import warnings 
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS   = ["SPY", "QQQ", "AAPL"]
TEST_SIZE = 500
LAGS =[1,2,3,5,10,21]

def load_processed(ticker : str)-> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}. Run Day 2 first.")
    return pd.read_csv(path, index_col="Date", parse_dates=True)

def return_vol_correlations(df : pd.DataFrame , ticker:str)-> pd.DataFrame:

    rv_1d = df["log_return"]**2*252
    r = df["log_return"]
    rows = []

    for k in LAGS:
        r_lag = r.shift(k)
        rv_lag = rv_1d.shift(k)

        combined = pd.concat([rv_1d , r_lag , rv_lag] , axis = 1).dropna()
        combined.columns=["rv" , "r_lag" , "rv_lag"]

        corr_r_rv = combined["rv"].corr(combined["r_lag"])
        corr_r2_rv = combined["rv"].corr(combined["r_lag"]**2)
        corr_abs_rv = combined["rv"].corr(combined["r_lag"].abs())

        n = len(combined)
        def corr_tstat(r_val):
            if abs(r_val)>=1: return np.nan
            return r_val * np.sqrt(n-2) / np.sqrt(1-r_val**2)
        
        rows.append({
            "Ticker"         : ticker,
            "Lag"            : k,
            "corr_r_rv"      : round(corr_r_rv,   4),
            "corr_r2_rv"     : round(corr_r2_rv,  4),
            "corr_abs_rv"    : round(corr_abs_rv, 4),
            "tstat_r_rv"     : round(corr_tstat(corr_r_rv), 3),
            "n"              : n,
        })

    return pd.DataFrame(rows)

def asymmetric_regression(df: pd.DataFrame , ticker:str)-> pd.DataFrame:
    rv_1d = (df["log_return"]**2*252)
    r = df["log_return"]

    rows = []
    for k in [1,5,21]:
        r_lag = r.shift(k)
        r_pos = r_lag.clip(lower=0)
        r_neg = r_lag.clip(upper = 0)
        rv_next = rv_1d.shift(-1)

        combined = pd.concat([rv_next, r_pos , r_neg] , axis = 1).dropna()
        combined.columns = ["rv_next" , "r_pos" , "r_neg"]

        X = combined[["r_pos" , "r_neg"]].values
        y = combined["rv_next"].values

        reg = LinearRegression().fit(X,y)
        b_pos , b_neg = reg.coef_
        
        y_pred = reg.predict(X)
        resid = y-y_pred
        sse = np.sum(resid**2)
        n , p = len(y), 2
        mse = sse / (n-p-1)

        #Variance of the b_pos and b_neg 
        XtX_inv = np.linalg.pinv(X.T @ X)
        c = np.array([1,-1])
        var_diff = mse*(c@XtX_inv @ c)
        t_asym = (b_pos - b_neg) / np.sqrt(max(var_diff , 1e-20))
        p_asym = 2*(1- stats.t.cdf(abs(t_asym) , df = n-p-1))

        rows.append({
            "Ticker" : ticker, 
            "Lag" : k, 
            "beta_pos" : round(b_pos , 6), 
            "beta_neg" : round(b_neg , 6),
            "asymmetry" : round(b_neg - b_pos , 6), 
            "t_asym" : round(t_asym , 4),
            "p_asym" : round(p_asym , 4), 
            "leverage_confirmed" : bool(b_neg < b_pos and p_asym < 0.05), 
            "R2" : round(reg.score(X , y) , 4),
        })

    return pd.DataFrame(rows)

def engle_ng_sign_bias_test(df: pd.DataFrame , ticker : str)-> dict:
    rv_1d = df["log_return"]**2*252
    r = df["log_return"]

    #Standardised by rolling 21-day vol as a Simple GARCH proxy 
    roll_vol = r.rolling(21, min_periods=10).std()
    eps = (r / roll_vol + 1e-10).dropna()

    # Sign indicators on lagged standardisation innovation


        
    eps_lag = eps.shift(1)
    S_neg = (eps_lag < 0).astype(float)
    S_pos = (eps_lag>=0).astype(float)
    interaction_neg = S_neg * eps_lag
    interaction_pos = S_pos * eps_lag

    # Auxiliary regression: eps_t^2 ~ S_neg + neg_interaction + pos_interaction
    eps_sq = eps**2
    combined = pd.concat([eps_sq , S_neg , interaction_neg , interaction_pos], axis = 1).dropna()
    combined.columns = ["eps_sq" , "S_neg" , "neg_int" , "pos_int"]

    X = combined[["S_neg" , "neg_int" , "pos_int"]].values
    y = combined["eps_sq"].values

    reg = LinearRegression().fit(X,y)
    y_pred = reg.predict(X)
    ss_res = np.sum((y-y_pred)**2)
    ss_total = np.sum((y-y.mean())**2)
    r2 = 1 - ss_res / ss_total

    n = len(y)
    lm = n * r2 # LM Statistic
    p_val = 1 - stats.chi2.cdf(lm , df=3) # chi_squared (3)

    coefs = dict(zip(["S_neg" , "neg_int" , "pos_int"] , reg.coef_))

    return {
        "Ticker" : ticker , 
        "LM_Stat" : round(lm , 4),
        "p_value" : round(p_val , 4), 
        "reject_symmetry" : bool(p_val < 0.05), 
        "coef_S_neg" : round(coefs["S_neg"] , 6), 
        "coef_neg_int" : round(coefs["neg_int"] , 6), 
        "coef_pos_int" : round(coefs["pos_int"] , 6), 
        "n" : n,

    }

def run_leverage_analysis()-> dict:
    """Run all leverage effect tests for all tickers."""
    print(f"\n{'='*55}")
    print("  DAY 17 — Leverage Effect Analysis")
    print(f"{'='*55}")

    all_corr = []
    all_asym = []
    all_ng = []
    results = {}

    for ticker in TICKERS:
        print(f"\n  {ticker}:")
        df = load_processed(ticker)

        corr_df = return_vol_correlations(df , ticker)
        asym_df = asymmetric_regression(df , ticker)
        ng_dict = engle_ng_sign_bias_test(df , ticker)

        all_corr.append(corr_df)
        all_asym.append(asym_df)
        all_ng.append(pd.DataFrame([ng_dict]))

        ## Printing the summary 
        neg_corr = corr_df[corr_df["Lag"]==1]["corr_r_rv"].values[0]
        lev_conf = asym_df[asym_df["Lag"]==1]["leverage_confirmed"].values[0]
        print(f"Lag-1 corr(r , future_vol) : {neg_corr:.4f}")
        print(f"Leverage confirmed Lag-1 : {lev_conf}")
        print(f"Engle-Ng LM stat : {ng_dict['LM_Stat']:.4f}"
              f" p = {ng_dict['p_value']:.4f} "
              f"{'REJECT symmetry' if ng_dict['reject_symmetry'] else 'cannot reject'}")
        
        results[ticker] = {
            "df" : df, 
            "corr_df" : corr_df, 
            "asym_df" : asym_df, 
            "ng" : ng_dict,
        }

    #Saving the results after all tickers are processed
    corr_out = pd.concat(all_corr , ignore_index = True)
    asym_out = pd.concat(all_asym , ignore_index = True)
    ng_out = pd.concat(all_ng , ignore_index = True)

    corr_out.to_csv(os.path.join(OUT_DIR, "day17_return_vol_corr.csv"),
                index=False)
    asym_out.to_csv(os.path.join(OUT_DIR, "day17_asymmetric_regression.csv"),
                index=False)
    ng_out.to_csv(  os.path.join(OUT_DIR, "day17_engle_ng_test.csv"),
                index=False)
    print(f"\n  [OK] day17_return_vol_corr.csv")
    print(f"  [OK] day17_asymmetric_regression.csv")
    print(f"  [OK] day17_engle_ng_test.csv")

    return results


if __name__ == "__main__":
    results = run_leverage_analysis()



