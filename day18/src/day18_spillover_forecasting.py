import os 
import warnings 
warnings.filterwarnings("ignore")
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


import numpy as np 
import pandas as pd 
from sklearn.linear_model import Ridge 
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline 

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS = {"SPY" , "QQQ" , "AAPL"}
TEST_SIZE = 500 
ALPHA = 10.0 

def load_rv_panel()-> pd.DataFrame: 
    series = {}
    for ticker in TICKERS:
        path = os.path.join(DATA_DIR , f"{ticker}_processed.csv")
        df = pd.read_csv(path , index_col = "Date" , parse_dates = True)
        rv = df["log_return"]**2*252
        series[ticker] = rv.rename(ticker)
    return pd.concat(series , axis = 1).dropna(how = "any")

def build_spillover_features(rv_df: pd.DataFrame, target_col : str, tci_series: pd.Series = None)-> tuple:
    target = rv_df[target_col]
    others = [c for c in rv_df.columns if c != target_col]
    features = pd.DataFrame(index=rv_df.index)

    # Set A : own HAR features
    features["rv_lag_1"] = target.shift(1)
    features["rv_lag_5"] = target.shift(1).rolling(5 , min_periods=5).mean()
    features["rv_lag_21"] = target.shift(1).rolling(21 , min_periods = 21).mean()



    # Set B : Cross - Asset lags
    for other in others: 
        features[f"cross_{other}_lag1"] = rv_df[other].shift(1)
        features[f"cross_{other}_lag5"] = rv_df[other].shift(1).rolling(5 , min_periods = 5).mean()

        # Relative Spillover : how elevated is other vs target?
        ratio = (rv_df[other].shift(1) / (target.shift(1) + 1e-10)).clip( 0 , 10)
        features[f"ratio_{other}_lag1"] = ratio

    #Set C : TCI Features
    if tci_series is not None: 
        tci_aligned = tci_series.reindex(rv_df.index).ffill()
        features["tci_lag1"] = tci_aligned.shift(1)
        features["tci_lag5"] = tci_aligned.shift(1).rolling(5 , min_periods = 3).mean()
        features["tci_vol_21"] = tci_aligned.shift(1).rolling(21 , min_periods = 10 ).std()

        ## TCI Regime flag: above 75th Percentile = high connectedness 
        features["high_tci"] = (tci_aligned.shift(1) > tci_aligned.quantile(0.75)).astype(float)

    target_next = target.shift(-1)
    combined = pd.concat([features , target_next.rename("target")] , axis = 1).dropna()
    X = combined.drop(columns = ["target"])
    y = combined["target"]
    return X , y 

def run_model(X_tr : pd.DataFrame , y_tr : pd.Series , X_te : pd.DataFrame , col_set : list)-> np.ndarray: 
    cols_ok = [ c for c in col_set if c in X_tr.columns]
    pipe = Pipeline([("sc" , StandardScaler()) , ("ridge" , Ridge(alpha = ALPHA))])
    pipe.fit(X_tr[cols_ok].fillna(0), y_tr)
    return pipe.predict(X_te[cols_ok].fillna(0))

def run_spillover_forecasting(connectedness : dict)-> dict: 
    print(f"\n{'='*55}")
    print("  DAY 18 — Spillover-Augmented Forecasting")
    print(f"{'='*55}")

    rv_df = load_rv_panel()
    ##Load rolling TCI if available 
    tci_path = os.path.join(OUT_DIR, "day18_rolling_tci.csv")
    tci_series = None
    if os.path.exists(tci_path):
        tci_df     = pd.read_csv(tci_path, index_col=0, parse_dates=True)
        tci_series = tci_df["TCI"]
        print(f"  TCI loaded: {len(tci_series)} observations")
    else:
        print("  ⚠ TCI file not found — skipping Set C features")

    all_metrics = []

    for target in TICKERS:
        print(f"\n  {target}:")
        others = [t for t in TICKERS if t != target]

        X, y   = build_spillover_features(rv_df, target, tci_series)
        n      = len(X)
        split  = n - TEST_SIZE

        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y.iloc[:split], y.iloc[split:]

        # Column sets for each model
        har_cols  = ["rv_lag_1","rv_lag_5","rv_lag_21"]
        cross_cols= har_cols + [f"cross_{o}_lag1" for o in others] + \
                    [f"cross_{o}_lag5" for o in others] + \
                    [f"ratio_{o}_lag1" for o in others]
        tci_cols  = cross_cols + ["tci_lag1","tci_lag5",
                                   "tci_vol_21","high_tci"]

        y_v = y_te.values
        for name, cols in [("HAR",          har_cols),
                            ("HAR+Spillover",cross_cols),
                            ("HAR+TCI",      tci_cols)]:
            preds = run_model(X_tr, y_tr, X_te, cols)
            rmse  = np.sqrt(np.mean((y_v - preds)**2))
            qlike = float(np.mean(
                np.log(np.maximum(preds,1e-8)) +
                np.maximum(y_v,1e-8)/np.maximum(preds,1e-8)
            ))
            print(f"    {name:18s}: RMSE={rmse:.6f}  QLIKE={qlike:.4f}")
            all_metrics.append({
                "Ticker": target, "Model": name,
                "RMSE"  : round(rmse, 8), "QLIKE": round(qlike, 6),
            })

    met_df = pd.DataFrame(all_metrics)
    met_df.to_csv(
        os.path.join(OUT_DIR, "day18_spillover_metrics.csv"), index=False
    )
    print(f"\n  ✓ day18_spillover_metrics.csv")

    print("\n  RMSE improvement: HAR+Spillover vs HAR:")
    pivot = met_df.pivot_table(index="Model", columns="Ticker",
                                values="RMSE") * 1e4
    print(pivot.round(2).to_string())

    return {"metrics": met_df}


