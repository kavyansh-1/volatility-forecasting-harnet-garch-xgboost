import os 
import warnings 
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd 
from sklearn.linear_model import Ridge 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS   = ["SPY", "QQQ", "AAPL"]
TEST_SIZE = 500
ALPHA_RIDGE = 10.0

def build_ahar_features(df: pd.DataFrame)-> tuple:
    rv_1d = df["log_return"] ** 2 * 252
    r = df["log_return"]

    rv_pos = (r.clip(lower=0) ** 2)*252
    rv_neg = (r.clip(upper = 0)**2)*252

    X = pd.DataFrame(index = df.index)

    X["rv_lag_1"] = rv_1d.shift(1)
    X["rv_lag_5"] = rv_1d.shift(1).rolling(5 , min_periods = 5).mean()
    X["rv_lag_21"] = rv_1d.shift(1).rolling(21 ,  min_periods = 21).mean()

    #AHAR : split daily component into up/down
    X["rv_pos_lag_1"] = rv_pos.shift(1)
    X["rv_neg_lag_1"] = rv_neg.shift(1)

    #AHAR : weekly/monthly: upside and downside averages 
    X["rv_pos_lag_5"] = rv_pos.shift(1).rolling(5 , min_periods = 5).mean()
    X["rv_neg_lag_5"] = rv_neg.shift(1).rolling(5 , min_periods = 5).mean()
    X["rv_pos_lag_21"] = rv_pos.shift(1).rolling(21 , min_periods = 21).mean()
    X["rv_neglag_21"] = rv_pos.shift(1).rolling(21 , min_periods = 21).mean()
    
    #Leverage term: r_{t-1} * I(r_{t-1} < 0)
    #This is negative on down days → with gamma<0, adds to vol forecast
    X["leverage_term"] = (r*(r<0).astype(float)).shift(1)
    #Return symmetry over 5 day window
    X["ret_asym_5"] = (r.shift(1).rolling(5 , min_periods = 3).apply(lambda x : (x[x<0]).sum() - (x[x<0]).sum()))
    target = rv_1d.shift(-1)
    combined = pd.concat([X.target.rename("target")] , axis = 1).dropna()
    return combined.drop(columns = ["target"]) , combined ["target"]

def fit_har_variants(X : pd.DataFrame , y: pd.Series , ticker : str)-> dict:
    n = len(X)
    split = n -TEST_SIZE
    X_tr , X_te = X.iloc[:split], X.iloc[split:]
    y_tr , y_te = y.iloc[:split] , y.iloc[split:]

    col_sets = {
        "HAR" : ["rv_lag_1" , "rv_lag_5" , "rv_lag_21"] , 
        "AHAR" : ["rv_pos_lag_1" , "rv_neg_lag_1" , "rv_pos_lag_5" , "rv_neg_lag_5" , "rv_lag_21"], 
        "L-HAR" : ["rv_lag_1" , "rv_lag_5" , "rv_lag_21" , "leverage_term"],
        "Full-AHAR" : ["rv_pos_lag_1" , "rv_neg_lag_1", "rv_pos_lag_5" , "rv_neg_lag_5" , "rv_lag_21" , "leverage_term" , "ret_asym_5"], 


    }

    predictions = {}
    metrics = []

    for name , cols in col_sets.items():
        cols_avail = [c for c in cols if c in X_tr.columns]
        if len(cols_avail) < 2:
            continue

        pipe = Pipeline([
            ("sc" , StandardScaler()), 
            ("ridge" , Ridge(alpha = ALPHA_RIDGE)),

        ])
        pipe.fit(X_tr[cols_avail].fillna(0), y_tr)
        pred = pipe.predict(X_te[cols_avail].fillna(0))
        predictions[name] = pred

        rmse = np.sqrt(np.mean((y_te.values - pred)**2))
        qlike = float(np.mean(np.log(np.maximum(pred , 1e-8)) + np.maximum(y_te.values,1e-8)/ np.maximum(pred , 1e-8)))

        #Coeffecient on Leverage term (if present)
        lev_coef = None
        if "leverage_term" in cols_avail:
            idx = cols_avail.index("leverage_term")
            sc = pipe.named_steps["sc"]
            rd = pipe.named_steps["ridge"]
            lev_coef = round(float(rd.coef_[idx]) , 6)

        #Asymmetry in pos/neg daily coeffecients
        asym_coef = None
        if "rv_neg_lag_1" in cols_avail and "rv_pos_lag_1" in cols_avail:
            sc = pipe.named_steps["sc"]
            rd = pipe.named_steps["ridge"]
            idx_n = cols_avail.index("rv_neg_lag_1")
            idx_p = cols_avail.index("rv_pos_lag_1")
            asym_coef = round(float(rd.coef_[idx_n] - rd.coef_[idx_p]) , 6)

        metrics.append({
            "Ticker" : ticker , 
            "Model" : name, 
            "RMSE" : round(rmse , 8), 
            "QLIKE" : round(qlike , 6), 
            "lev_coef" : lev_coef, 
            "asym_coef" : asym_coef, 

        })

        print(f" {name:12s}:  RMSE={rmse:.6f} QLIKE = {qlike:.4f}"
              + (f" lev_coef = {lev_coef:.4f}" if lev_coef else"")
              + (f" asym = {asym_coef:.4f}" if asym_coef else""))
    
    return {
        "predictions" : predictions, 
        "y_test" : y_te, 
        "metrics" : pd.DataFrame(metrics),

    }

def run_asymmetric_har() -> dict:
    """Fit all asymmetric HAR variants for all tickers."""
    print(f"\n{'='*55}")
    print("  DAY 17 — Asymmetric HAR Models")
    print(f"{'='*55}")

    results  = {}
    all_mets = []

    for ticker in TICKERS:
        path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
        if not os.path.exists(path):
            continue

        print(f"\n  {ticker}:")
        df   = pd.read_csv(path, index_col="Date", parse_dates=True)
        X, y = build_ahar_features(df)
        res  = fit_har_variants(X, y, ticker)
        results[ticker] = res
        all_mets.append(res["metrics"])

    if all_mets:
        met_df = pd.concat(all_mets, ignore_index=True)
        met_df.to_csv(
            os.path.join(OUT_DIR, "day17_ahar_metrics.csv"), index=False
        )
        print(f"\n  ✓ day17_ahar_metrics.csv")
        print("\n  Summary (RMSE × 10^4):")
        pivot = met_df.pivot_table(
            index="Model", columns="Ticker", values="RMSE"
        ) * 1e4
        print(pivot.round(2).to_string())

    return results
