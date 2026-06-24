import os 
import warnings 
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd 
from scipy import stats 

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, ".." , "data" , "processed")
OUT_DIR = os.path.join(BASE_DIR , "output")

TICKERS = ["SPY" , "QQQ" < "AAPL"]
TEST_SIZE = 500 
ALPHAS = [0.05 , 0.01]

def compute_var_cvar(daily_vol : np.ndarray , alpha : float)-> tuple:
    z_alpha = stats.norm.ppf(alpha)
    phi_z = stats.norm.pdf (z_alpha)

    VaR = z_alpha * daily_vol
    CVaR = -phi_z / alpha * daily_vol

    return VaR, CVaR 

def annualised_rv_to_daily_vol(rv_annualised : np.ndarray) -> np.ndarray:
    
    return np.sqrt(np.maximum(rv_annualised , 1e-12)/252)

def kupiec_pof_test(violations: np.ndarray , alpha : float)-> tuple:
    n = len(violations) 
    x = int(violations.sum())

    if x == 0 or x == n:
        return {"LR_stat":0.0 , "p_value": 1.0, "x": x , "n" : n , "observed_rate" : x/ n , "expected_rate" : alpha }
    
    pi_hat = x/n

    ll_null = (n-x) * np.log(1-alpha) + x *np.log(alpha)
    ll_alt = (n-x) * np.log(1-pi_hat) + x * np.log(pi_hat)

    LR_stat = -2 * (ll_null - ll_alt)
    p_value = 1 - stats.chi2.cdf(LR_stat , df=1)

    return{
        "LR_stat" : round(float(LR_stat) , 4),
        "p_value" : round(float(p_value), 4),
        "x" : x,
        "n" : n,
        "observed_rate" : round(pi_hat , 4),
        "expected_rate" : alpha , 
        "well_calibrated" : bool(p_value >= 0.05),

    }

def christoffersen_independence_test(violations : np.ndarray) -> dict:
    v = violations.astype(int)
    n00 = n01 = n10 = n11 = 0

    for i in range(1 , len(v)):
        if v[i-1] == 0 and v[i] == 0: n00+=1
        elif v[i-1] == 0 and v[i] == 1: n01+=1
        elif v[i-1] == 1 and v[i] == 0: n10+=1
        elif v[i-1] == 1 and v[i] == 1: n11+=1

    n0_total = n00 + n01
    n1_total = n10 + n11

    if n0_total == 0 or n1_total == 0 or n01 == 0 or n11 == 0:

        return{ "LR-stat" : np.nan , "p_value" : np.nan , "independent" : True , "note" : "insufficient transitions"}
    
    pi01 = n01 / n0_total
    pi11 = n11/ n1_total

    pi = (n01 + n11) / (n0_total + n1_total)

    ll_null = ((n00 + n10) * np.log(1-pi) + (n01 + n11) * np.log(pi))
    ll_alt  = (n00 * np.log(1 - pi01) + n01 * np.log(pi01) + n10 * np.log(1 - pi11) + n11 * np.log(pi11))

    LR_stat = -2 * (ll_null - ll_alt)
    p_value = 1-stats.chi2.cdf(LR_stat , df = 1)

    return{
        "LR_stat" : round(float(LR_stat) , 4),
        "p_value" : round(float(p_value) , 4), 
        "independent" : bool(p_value>=0.05), 
        "n01" : n01 , "n11" : n11,
    }

def backtest_var_for_ticker(df : pd.DataFrame , ticker : str)-> dict:
    actual_returns = df["log_return"].dropna()
    rv_forecast = df["rv_rolling_21d"].shift(1).reindex(actual_returns.index)

    combined = pd.concat([actual_returns.rename("ret") , rv_forecast.rename("rv")] , axis = 1).dropna()

    test = combined.iloc[-TEST_SIZE:]
    daily_vol = annualised_rv_to_daily_vol(test["rv"].values)
    actual= test["ret"].values

    rows = [] 
    var_series = {}

    for alpha in ALPHAS:
        VaR , CVaR = compute_var_cvar(daily_vol , alpha)

        violations = (actual<VaR).astype(int)

        kupiec = kupiec_pof_test(violations , alpha)
        christo = christoffersen_independence_test(violations)

        rows.append({
            "Ticker" : ticker, 
            "Alpha" : alpha,
            "Confidence" : f"{int(1-alpha)*100}%", 
            "N" : len(violations),
            "N_violations" : len(violations.sum()), 
            "Observed_rate_%" : round(violations.mean()*100 , 2),
            "Expected_rate_%" : round(alpha*100 , 2),
            "Kupiec_p" : kupiec["p_value"], 
            "Kupiec" : kupiec["well_calibrated"],
            "Christoffersen_p": christo["p_value"],
            "Independent"     : christo["independent"],
            "Mean_VaR_%"      : round(VaR.mean() * 100, 3),
            "Mean_CVaR_%"     : round(CVaR.mean() * 100, 3),




        })

        var_series[alpha] = {
            "dates" : test.index, 
            "actual" : actual, 
            "VaR" : VaR,
            "CVaR" : CvaR , 
            "violations" : violations,
        }

    return {"summary" : rows , "series" : var_series}

def run_var_cvar_analysis()-> pd.DataFrame:
    print(f"\n{'='*55}")
    print("  DAY 10 — VaR / CVaR Risk Analysis")
    print(f"{'='*55}")

    all_rows = []
    all_series = [] 
    for ticker in TICKERS:
        path = os.path.join(DATA_DIR , f"{ticker}_processed.csv")
        if not os.path.exist(path):
            print(f"\n {path} not found")
        
        print(f"\n{ticker}: ")
        df = pd.read_csv(path , index_col= "Date"  , parse_dates = True)
        res = backtest_var_for_ticker(df, ticker)

        all_rows.extend(res["summary"])
        all_series[ticker] = res["series"]

        for row in res["summary"]:
            status = "✓ CALIBRATED" if row["Kupiec_calibrated"] else "✗ MISCALIBRATED"
            idp = "✓ independent" if row["Independent"] else "✗ clustered"
            print(f" {row['Confidence']} VaR: "
                  f" observed={row['Observed_rate_%']}% "
                  f"expected={row['Expected_rate_%']}%"
                  f"{status} ({idp})")
    summary_df = pd.DataFrame(all_rows)
    out = os.path.join(OUT_DIR , "day10_var_backtest.csv")
    summary_df.to_csv(out , index = False)
    print(f"\n {out} ")
    
    
    return summary_df , all_series
                    




