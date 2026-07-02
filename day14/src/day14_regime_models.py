import os 
import sys 
import warnings 
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd 
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR , "..", "data" , "processed")
OUT_DIR = os.path.join(BASE_DIR , "output")
os.makedirs(OUT_DIR , exist_ok=True)

TICKERS = ["SPY" , "QQQ" , "AAPL"]
TEST_SIZE = 500
ALPHA_HAR = 10.0

REGIME_NAMES = {0: "Low" , 1: "Medium" , 2: "High"}

def build_base_features(df:pd.DataFrame)-> tuple:
    """ uses the same HAR features which I used like the rv_lag1 , rv_lag_5 , rv_lag_21 and in addition of ret lags and rolling vols.."""
    rv_1d = df["log_return"]**2*252
    feats = pd.DataFrame(index = df.index)

    for k in [1,2,3,5,10,21]:
        feats[f"rv_lag_{k}"] = rv_1d.shift(k)
    for k in [1,2,3,5]:
        feats[f"ret_lag_{k}"] = df["log_return"].shift(k)
    for col in ["rv_rolling_5d" , "rv_rolling_21d" , "rv_rolling_63d"]:
        if col in df.columns:
            feats[col] = df[col].shift(1)
    feats["rv_ewm_10"] = rv_1d.shift(1).ewm(span=10 , min_periods = 5).mean()
    feats["abs_ret_lag1"] = df["log_return"].abs().shift(1)

    target=rv_1d.shift(-1)
    combined = pd.concat([feats , target.rename("target")],axis = 1).dropna()
    return combined.drop(columns=["target"]), combined["target"]

def add_regime_features(X: pd.DataFrame , regime_ser: pd.Series, posteriors: pd.DataFrame)-> pd.DataFrame:
    X = X.copy()
    # This is the hard regime label from the previous day which will be used ahead 
    regime_lag = regime_ser.shift(1).reindex(X.index)
    X["regime_label"] = regime_lag.fillna(1) #kept as default

    for col in ["post_low" , "post_med" , "post_high"]:
        if col in posteriors.columns:
            X[f"lag_{col}"] = posteriors[col].shift(1).reindex(X.index).fillna(1/3)

    return X

def train_regime_conditional_har(X_tr : pd.DataFrame, y_tr : pd.Series , regime_col : str = "regime_label")-> dict:
    har_cols = [c for c in ["rv_lag_1" , "rv_lag_5" , "rv_lag_21"]
                if c in X_tr.columns]
    models = {}
    full_pipe = Pipeline([("sc" , StandardScaler()), ("ridge" , Ridge(alpha = ALPHA_HAR))])
    full_pipe.fit(X_tr[har_cols] , y_tr)
    models["fallback"] = full_pipe
    
    for regime_id in [0,1,2]:
        mask = (X_tr[regime_col] == regime_id)
        n = mask.sum()

        if n < 50:
            print(f" Regime {REGIME_NAMES[regime_id]}: only {n} samples - using full-data fallback")
            models[regime_id] = full_pipe

        else:
            pipe = Pipeline([("sc", StandardScaler()) , 
                             ("ridge", Ridge(alpha=ALPHA_HAR))])
            pipe.fit(X_tr.loc[mask, har_cols] , y_tr[mask])
            models[regime_id] = pipe
            print(f" Regime {REGIME_NAMES[regime_id]}: fitted on {n} samples")
            
    return models 

def predict_regime_conditional (X_te: pd.DataFrame , models : dict, regime_col : str = "regime_label")->np.ndarray:
    har_cols = [c for c in ["rv_lag_1" , "rv_lag_5" , "rv_lag_21"] if c in X_te.columns]
    preds = np.zeros(len(X_te))
    for i , (idx , row) in enumerate (X_te.iterrows()):
        regime_id = int(row.get(regime_col,1))
        model = models.get(regime_id , models["fallback"])
        x_row = row[har_cols].values.reshape(1 , -1)
        preds[i] = model.predict(x_row)[0]
    return preds 

def train_regime_augmented_xgb(X_tr: pd.DataFrame , y_tr: pd.Series)-> XGBRegressor:
    """ XGBoost model but with regime posterior probs as extra features 
    and it discovers the regime interactions automatically through splits..."""

    model = XGBRegressor(
        n_estimators = 200, 
        max_depth = 3, 
        learning_rate = 0.05, 
        subsample = 0.8, 
        colsample_bytree = 0.8,
        reg_lambda = 2.0, 
        objective = "reg:squarederror", 
        tree_method = "hist", 
        random_state = 42, 
        verbosity = 0, 

    )

    model.fit(X_tr.fillna(0), y_tr)
    return model 

def run_regime_models(hmm_results : dict)-> dict:
    print(f"\n{'='*55}")
    print("  DAY 14 — Regime-Conditional Models")
    print(f"{'='*55}")

    all_results = {}
    summary_rows = []

    for ticker in TICKERS:
        if ticker not in hmm_results:
            continue

        print(f"\n{ticker}: ")

        path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col="Date", parse_dates=True)

        X_base , y = build_base_features(df)
        regime_ser = hmm_results[ticker]["regime_series"]
        posteriors = hmm_results[ticker]["posteriors"]
        
        """ adding the regime features here now """
        X_aug = add_regime_features(X_base , regime_ser , posteriors)

        """ now alignment of all on the common index"""
        common = X_aug.index.intersection(y.index)
        X_aug = X_aug.loc[common]
        X_base = X_base.loc[common]
        y = y.loc[common]

        n = len(X_aug)
        split = n - TEST_SIZE

        X_aug_tr , X_aug_te = X_aug.iloc[:split] , X_aug.iloc[split:]
        X_base_tr , X_base_te = X_base.iloc[:split] , X_base.iloc[split:]
        y_tr , y_te = y.iloc[:split] , y.iloc[split:]

        # Model A: Baseline Global HAR ( this has no regime info)
        har_cols = [c for c in ["rv_lag_1", "rv_lag_5" , "rv_lag_21"]
                    if c in X_base_tr.columns]
        pipe_global = Pipeline([("sc" , StandardScaler()), 
                                ("ridge", Ridge(alpha=ALPHA_HAR))])
        pipe_global.fit(X_base_tr[har_cols], y_tr)
        pred_global = pipe_global.predict(X_base_te[har_cols])

        # Now my Model-B Regime Conditional HAR 
        print("Training my regime conditional HAR....")
        regime_models = train_regime_conditional_har(X_aug_tr , y_tr)
        pred_regime_har = predict_regime_conditional(X_aug_te , regime_models)

        # Now the last but not the least Regime Augmented XGBoost
        print("Training regime augmented XGBoost....")
        xgb_model = train_regime_augmented_xgb(X_aug_tr , y_tr)
        pred_xgb = xgb_model.predict(X_aug_te.fillna(0))

        all_results[ticker] = {
            "y_test" : y_te.values, 
            "pred_global_har" : pred_global, 
            "pred_regime_har" : pred_regime_har,
            "pred_xgb_regime" : pred_xgb,
            "X_test" : X_aug_te, 
            "regime_test" : X_aug_te.get("regime_label" , pd.Series()).values, 
            "dates_test": X_aug_te.index,
        } 

    # For saving predictions
    rows = []
    for ticker , res in all_results.items():
        for i, (y , gh , rh , xg) in enumerate(zip(
            res["y_test"] , res["pred_global_har"] , res["pred_regime_har"] , res["pred_xgb_regime"]
        )):
            rows.append({
                "ticker" : ticker,
                "actual" : y, 
                "pred_global_har" : gh, 
                "pred_regime_har": rh,
                "pred_xgb_regime" : xg, 
                "regime" : int(res["regime_test"][i]) if i < len(res["regime_test"]) else -1,
            })

    pd.DataFrame(rows).to_csv(
        os.path.join(OUT_DIR, "day14_regime_predictions.csv"), index = False
    )
    print(f"\n✓ day14_regime_predictions.csv ")
    return all_results








 





    
