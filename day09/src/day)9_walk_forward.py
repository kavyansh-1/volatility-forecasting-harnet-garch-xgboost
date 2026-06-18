import os 
import sys 
import warnings 
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd 
from sklearn.linear_model import Ridge 
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR , ".." , "data" , "processed")
OUT_DIR = os.path.join(BASE_DIR , "output")
os.makedirs(OUT_DIR , exist_ok  = True)

TICKERS = ["SPY" , "QQQ" , "AAPL"]
INITIAL_TRAIN = 756
STEP_SIZE = 21
ROLLING_WINDOW = 756

def build_features_and_target(df:pd.DataFrame)-> tuple:
    rv_1d = df["log_return"] ** 2 * 252
    feats = pd.DataFrame(index = df.index)
    for k in [1,2,3,5,10,21]:
        feats[f"ret_lag_{k}"] = rv_1d.shift(k)

    for k in [1,2,3,5]:
        feats[f"ret_lag_{k}"] = df["log_return"].shift(k) 

    for col in ["rv_rolling_5d" , "rv_rolling_21d" , "rv_rolling_21d"]:
        if col in df.columns:
            feats[col]=df[col].shift(1)
    
    for col in ["park_vol_5d" , "gk_vol_5d"]:
        if col in df.columns:
            feats[col] = df[col].shift(1)
    
    feats["rv_ewm_10"] = rv_1d.shift(1).ewm(span = 10 , min_periods = 5).mean()
    feats["abs_ret_lag1"] = df["log_return"].abs().shift(1)

    target = rv_1d.shift(-1)
    combined =  pd.concat([feats, target.rename("target")] , axis=1).dropna()

    X = combined.drop(columns = ["target"])
    y = combined["target"]
    return X , y

def _metrics(y_true: np.ndarray , y_pred: np.ndarray) -> dict:
    eps = 1e-8
    h = np.maximum(y_pred , eps)
    v = np.maximum(y_true , eps)
    rmse = np.sqrt(np.mean((y_true - y_pred)** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    ql = np.mean(np.log(h) + v / h)
    dacc = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100 if len(y_true) > 1  else np.nan
    return {"RMSE" : rmse , "MAE" : mae , "QLIKE" : ql, "DirAcc": dacc }


def har_walk_forward( X: pd.DataFrame, y: pd.Series, alpha: float = 12.0, mode: str = "expanding" , label : str = "HAR") -> pd.DataFrame:
  n = len(X)
  rows = []

  har_cols = [c for c in ["rv_lag_1" , "rv_lag_5" , "rv_lag_21"]
  if c in X.columns]
  X_har = X[har_cols]

  for window_idx , start in enumerate (
    range (INITIAL_TRAIN , n-STEP_SIZE , STEP_SIZE)
  ):
     end = min(start + STEP_SIZE , n)
     if mode == "expanding":
        tr_start = 0
     
     else:
        tr_start = max(0, start - ROLLING_WINDOW)

     X_tr = X_har.iloc[tr_start : start]
     y_tr = y.iloc[tr_start : start]
     X_te = X_har.iloc[start : end]
     y_te  = y.loc[start : end]

     if len(X_tr) < 30 or len(X_te) == 0:
        continue

     pipe = Pipeline([
        ("sc", StandardScaler()),
        (("ridge" , Ridge(alpha = alpha)))
     ])
     pipe.fit(X_tr , y_tr)
     preds = pipe.predict(X_te)

     m = _metrics(y_te.values , preds)
     rows.append({
        "window" : window_idx,
        "model" : label, 
        "mode" : mode,
        "train_start" : X.index[tr_start].date(),
        "train_end" : X.index[start-1].date(),
        "test_start" : X.index[start].date(), 
        "test_end" : X.index[end-1].date(),
        "n_train" : len(X_tr), **m, 
        "preds" : preds.tolist(), 
        "actuals" : y_te.values.tolist(),

     })

     return pd.DataFrame(rows)

def xgb_walk_forward(X: pd.DataFrame, y : pd.Series, params: dict = None , mode: str = "expanding" , label: str = "XGBoost")-> pd.DataFrame:

    if params is None: 
        params = {
            "n_estimators" : 200, 
            "max_depth" : 3,
            "learning_rate" : 0.05,
            "subsample" : 0.8,
            "colsample_bytree" : 0.8,
            "reg_lambda" : 2.0,


        }
     
    n = len (X)
    rows = []

    for window_idx , start in enumerate( 
        range(INITIAL_TRAIN, n- STEP_SIZE, STEP_SIZE)):
        end = min ( start + STEP_SIZE , n)

        tr_start = 0 if mode == "expanding" \
            else max ( 0 , start - ROLLING_WINDOW)

        X_tr = X.iloc[tr_start : start]
        y_tr = y.iloc[tr_start : start]
        X_te = x.iloc[start : end]
        y_te = y.iloc[start : end]

        if len(X_tr) < 50 or len (X_te) == 0:
            continue 

        model = XGBRegressor(
            objective = "reg:squarederror",
            tree_method = "hist",
            random_state = 42 , 
            verbosity = 0,
            **params,

        )
        model.fit(X_tr , y_tr)
        preds = model.predict(X_te)

        m = _metrics(y_te.values(), preds)
        rows.append({
            "window" : window_idx,
            "model" : model,
            "mode" : mode, 
            "train_start" : X.index[tr_start].date(),
            "train_end" : X.index[start-1].date(),
            "test_start" : X.index[start].date(),
            "test_end" : X.index[end-1].date(),
            "n_train" : len(X_tr), **m,
            "preds" : preds.tolist(),
            "actuals" : y_te.values.tolist(),
             })
    return pd.DataFrame(rows)

    def run_walk_forward_all(tickers : list = None) -> dict:

        print(f"\n{'='*55}")
        print("  DAY 9 — Walk-Forward Validation")
        print(f"{'='*55}")
        print(f"Initial train : {INITIAL_TRAIN} days")
        print(f"Step Size : {STEP_SIZE} days")
        print(f" Rolling Window : {ROLLING_WINDOW} days")

        if tickers is None: 
            tickers = TICKERS

        all_results = {}
        summary_rows = {}

        for ticker in tickers:
            path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
            if not os.path.exists(path):
                print(f"\n  ⚠ {path} not found — skipping")
                continue

            print(f"\n  {'─'*45}")
            print(f"  {ticker}")
            print(f"  {'─'*45}")

            df = pd.read_csv(path , index_col = "Date", parse_dates=True)
            X,y = build_features_and_target(df)

            ticker_results = {}

            experiments = [
                ("HAR_expanding",  "har",  "expanding"),
                ("HAR_rolling",    "har",  "rolling"),
                ("XGB_expanding",  "xgb",  "expanding"),
                ("XGB_rolling",    "xgb",  "rolling"),

            ]

            for exp_label, model_type , mode in experiments: 
                print(f"Running{exp_label}...")

                if model_type == "har":
                    res_df = har_walk_forward(X,y,mode=mode , label=exp_label)

                else:
                    res_df = xgb_walk_forward(X, y , mode = mode, label = exp_label)
                
                ticker_results[exp_label] = res_df

                if not res_df.empty:
                    mean_rmse = res_df["RMSE"].mean()
                    std_rmse = res_df["RMSE"].std()
                    n_windows = len(res_df)
                    print(f" Windows = {n_windows}"
                    f"mean RMSE {mean_rmse:.6f}",
                    f"std = {std_rmse:.6f}")

                    summary_rows.append({
                       "Ticker" : ticker,
                       "Experiment" : exp_label,
                       "N_windows" : n_windows,
                       "RMSE_mean" : round(mean_rmse , 8),
                       "RMSE_std" : round(std_rmse, 8),
                       "RMSE_min" : round(res_df["RMSE"].min() , 8),
                       "RMSE_max" : round(res_df["RMSE"].max() , 8),
                       "MAE_mean" : round(res_df["MAE"].mean(), 8),
                       "QLIKE_mean" : round(res_df["QLIKE"].mean(), 6),
                       "DirAcc_mean" : round(res_df["DirAcc"].mean(),2)


                    })
            all_results[ticker] = ticker_results
        
        summary_df = pd.DataFrame(summary_rows)
        out = os.path.join(OUT_DIR, "day09_walk_forward_summary.csv")
        summary_df.to_csv(out , index = False)
        print(f"\n  ✓ Walk-forward summary compleyed yayyaayaya → {out}")

        all_dfs = []
        for ticker , exps in all_results.items():
            for exp_label, df in exps.items():
                if not df.empty:
                    df_save = df.drop(columns = ["preds" , "actuals"], errors = "ignore").copy()
                    df_save.insert(0 , "Ticker" , ticker)
                    all_dfs.append(df_save)
        
        if all_dfs:
            combined = pd.concat(all_dfs , ignore_index = True)
            out2 = os.path.join(OUT_DIR , "day09_walk_forward_windows.csv")
            combined.to_csv(out2 , index = False)
            print(f"  ✓ Per-window results → {out2}")
                    
        return all_results




