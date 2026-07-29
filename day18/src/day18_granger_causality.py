import os 
import warnings 
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS   = ["SPY", "QQQ", "AAPL"]
MAX_LAGS = 5
ROLL_WINS = 252 ## Rolling window for the time-varying test

def load_rv_matrix()-> pd.DataFrame:
    series = {}
    for ticker in TICKERS:
        path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}. Run Day 2 first.")
        df = pd.read_csv(path , index_col = "Date" , parse_dates = True)
        rv = df["log_return"]**2*252
        series[ticker] = rv.rename(ticker)

    rv_df = pd.concat(series , axis = 1).dropna(how = "any")
    return rv_df

def build_lag_matrix(series : pd.Series , other : pd.Series , n_lags : int)-> tuple:
    data = pd.DataFrame({"y" : series , "x" : other})

    for k in range ( 1 , n_lags + 1):
        data[f"y_lag{k}"] = data["y"].shift(k)
        data[f"x_lag{k}"] = data["x"].shift(k)

    data = data.dropna()
    y = data["y"].values

    y_lags = [f"y_lag{k}" for k in range(1 , n_lags + 1)]
    x_lags = [f"x_lag{k}" for k in range(1 , n_lags + 1)]

    X_r = add_constant(data[y_lags].values)
    X_ur = add_constant(data[y_lags + x_lags].values)

    return X_r , X_ur , y , len(data)

def granger_f_test(series : pd.Series , other : pd.Series , n_lags : int, cause : str, effect: str)-> dict:

    X_r , X_ur , y , n = build_lag_matrix(series , other , n_lags)
    q = n_lags # number of restrictions (X lag coeffecients)
    k = X_ur.shape[1] - 1 ## regressors ( excl. intercept)

    #Fit both models via OLS
    res_r = OLS(y , X_r).fit()
    res_ur = OLS(y , X_ur).fit()

    rss_r = res_r.ssr
    rss_ur = res_ur.ssr

    if rss_ur < 1e-20:
        return {"cause" : cause , "effect" : effect, "n_lags" : n_lags , "F_stat" : np.nan , "p_value" : np.nan, "significant" : False , "n" : n}

    F_stat = ((rss_r - rss_ur) / q) / (rss_ur / (n-k-1))
    p_value = 1 - stats.f.cdf(F_stat , dfn = q , dfd = n-k-1)

    return{
        "cause" : cause, 
        "effect" : effect, 
        "n_lags" : n_lags, 
        "F_stat" : round(float(F_stat) , 4),
        "p_value" : round(float(p_value) , 4),
        "significant" : bool(p_value < 0.05), 
        "n" : n, 
        "R2_ur" : round(float(res_ur.rsquared) , 4), 
        "R2_r" : round(float(res_r.rsquared) , 4),
    }

def lag_selection_aic(series: pd.Series , other : pd.Series , max_lags : int = MAX_LAGS)-> int:
    best_aic = np.inf
    best_lags = 1

    for n_lags in range(1 , max_lags + 1):
        _, X_ur , y , n = build_lag_matrix(series , other , n_lags)
        res = OLS(y , X_ur).fit()
        aic = n*np.log(res.ssr / n) + 2 * X_ur.shape[1]
        if aic < best_aic:
            best_aic = aic
            best_lags = n_lags 

    return best_lags

def rolling_granger(series : pd.Series , other : pd.Series , cause : str , effect : str , n_lags : int = 1, win: int = ROLL_WINS)-> pd.DataFrame:
    n = len(series)
    rows = []

    for end in range(win , n):
        start = end - win
        s_win = series.iloc[start:end]
        o_win = other.iloc[start:end]

        try: 
            res = granger_f_test(s_win , o_win , n_lags , cause , effect)
            rows.append({
                "date" : series.index[end], 
                "F_stat" : res["F_stat"], 
                "p_value" : res["p_value"], 
                "significant" : res["significant"],


            })

        except Exception:
            pass

    return pd.DataFrame(rows)

def run_granger_causality()-> dict:
    print(f"\n{'='*55}")
    print("  DAY 18 - Granger Causality in Variance")
    print(f"{'='*55}")

    rv_df = load_rv_matrix()
    results = {"rv_df" : rv_df}
    all_rows = []
    roll_dfs = {}

    pairs = [
        ("SPY", "QQQ"), ("SPY", "AAPL"), ("QQQ", "AAPL"),
        ("QQQ", "SPY"), ("AAPL", "SPY"), ("AAPL", "QQQ"),
    ]

    print(f"\n  Full-sample Granger causality (RV proxy, F-test):")
    print(f"  {'Cause':8s} -> {'Effect':8s}  Lags  F-stat   p-val  Sig")
    print(f"  {'-'*52}")

    for cause, effect in pairs:
        best_lag = lag_selection_aic(rv_df[effect], rv_df[cause])
        res      = granger_f_test(rv_df[effect], rv_df[cause],
                                   best_lag, cause, effect)
        all_rows.append(res)
        sig = "*" if res["significant"] else " "
        print(f"  {cause:8s} -> {effect:8s}  "
              f"{best_lag:4d}  {res['F_stat']:7.3f}  "
              f"{res['p_value']:5.3f}  {sig}")

        # Rolling test
        roll_df = rolling_granger(rv_df[effect], rv_df[cause],
                                   cause, effect, n_lags=1)
        roll_dfs[f"{cause}_to_{effect}"] = roll_df

    # Save
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(
        os.path.join(OUT_DIR, "day18_granger_results.csv"), index=False
    )

    # Save rolling results
    for key, df in roll_dfs.items():
        df.to_csv(
            os.path.join(OUT_DIR, f"day18_rolling_granger_{key}.csv"),
            index=False
        )

    print(f"\n  [OK] day18_granger_results.csv")
    print(f"  [OK] day18_rolling_granger_{{pair}}.csv")

    results["granger_df"] = results_df
    results["roll_dfs"]   = roll_dfs
    return results

          





        