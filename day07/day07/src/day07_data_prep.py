import os 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler 

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DAY6_DIR = os.path.join(BASE_DIR, "..", "day06" , "output")
DATA_DIR = os.path.join(BASE_DIR , ".." , "data" , "processed")
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR , exist_ok = True)

TICKERS = ["SPY" , "QQQ" , "AAPL"]
TEST_SIZE = 500

SENT_COLS= [
    "sent_compound_mean",
    "sent_compound_std",
    "sent_pos_mean", 
    "sent_neg_mean",
    "sent_roll3", 
    "sent_roll7",
    "n_articles",
]

def build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    rv_1d = df["log_return"]**2*252
    feats = pd.DataFrame(index = df.index)

    for k in [1,2,3,5,10,21]:
        feats[f"rv_lag_{k}"] = rv_1d.shift(k)

    for k in [1,2,3,5]:
        feats[f"ret_lag_{k}"] = df["log_return"].shift(k)
    
    for col in ["rv_rolling_5d", "rv_rolling_21d" , "rv_rolling_63d"]:
        if col in df.columns:
            feats[col] = df[col].shift(1)
    
    for col in ["park_vol_5d" , "park_vol_21d" , "gk_vol_5d" , "gk_vol_21d"]:
        if col in df.columns:
            feats[col] = df[col].shift(1)
    
    if "rv_rolling_5d" in df.columns and "rv_rolling_21d" in df.columns:
        feats["rv_ratio_5_21"] = (
            df["rv_rolling_5d"].shift(1) / (df["rv_rolling_21d"].shift(1) + 1e-10)
            )

    feats["rv_ewm_10"] = rv_1d.shift(1).ewm(span = 10, min_periods = 5).mean()
    feats["abs_ret_lag1"] = df["log_return"].abs().shift(1)

    return feats

def build_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index = df.index)
    for col in SENT_COLS:
        if col in df.columns:
            feats[f"s_{col}"] = df[col].shift(1)

        if "sent_roll3" in df.columns:
            feats["s_sent_momentum"] = (
                df["sent_roll3"].shift(1) - df["sent_roll3"].shift(8)
            )
        
        if "sent_neg_mean" in df.columns and "rv_rolling_5d" in df.columns:
            feats["s_neg_x_vol"] = (
                df["sent_neg_mean"].shift(1) * df["rv_rolling_5d"].shift(1)
            )
        
        return feats


def prepare_dataset(ticker : str) -> dict:

    merged_path = os.path.join(
        DAY6_DIR, f"day06_{ticker}_sentiment_vol.csv"
    )
    day2_path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")

    if os.path.exists(merged_path):
        df = pd.read_csv(merged_path , index_col = "Date" , parse_dates = True)
        print(f"  {ticker}: loaded Day 6 merged file ({len(df)} rows)")
    elif os.path.exists(day2_path):
        df = pd.read_csv(day2_path, index_col = "Date" , parse_dates = True)
        print(f" {ticker}: Day 6 file missing - using Day 2 only")
    else:
        raise FileNotFoundError(
            f"No file found for {ticker}. Run Day 2 and Day 6 first"
        )

    rv_1d = df["log_return"]**2*252
    target = rv_1d.shift(-1)

    base_feats = build_baseline_features(df)
    sent_feats= build_sentiment_features(df)
    aug_feats = pd.concat([base_feats , sent_feats] , axis = 1).dropna()
    X = combined.drop(columns = ["target"])
    y = combined["target"]
    n = len(X)
    tr = n - TEST_SIZE
    return (X.iloc[:tr] , X.iloc[tr:],y.iloc[:tr], y.iloc[tr:], X.columns.tolist(), X.iloc[tr:].index)
(X_b_tr , X_b_te , y_tr , y_te , base_names , dates_te) = align_and_split(base_feats , target)
(X_a_tr, X_a_te, y_tr_, y_te_, aug_names, _) = align_and_split(aug_feats, target)

n_tr = min(len(X_b_tr), len(X_a_tr))
n_te = min(len(X_b_te) , len(X_a_te))

return {
    "ticker" : ticker,
    "X_base_train"    : X_b_tr.iloc[-n_tr:],
    "X_base_test"     : X_b_te.iloc[:n_te],
    "X_aug_train"     : X_a_tr.iloc[-n_tr:],
    "X_aug_test"      : X_a_te.iloc[:n_te],
    "y_train"         : y_tr.iloc[-n_tr:],
    "y_test"          : y_te.iloc[:n_te],
    "base_names"      : base_names,
    "aug_names"       : aug_names,
    "dates_test"      : dates_te[:n_te],
    "n_sent_features" : len(sent_feats.columns),

}

def prepare_all_tickers() -> dict:
    print(f"\n{'='*55}")
    print("DAY 7 - Data Preparation")
    print(f"\n{'='*55}")
    datasets = {}
    for ticker in TICKERS:
        datasets[ticker] = prepare_dataset(ticker)
        d = datasets[ticker]
        print(f"  {ticker}: train={len(d['y_train'])}  "
              f"test={len(d['y_test'])}  "
              f"base_feats={len(d['base_names'])}  "
              f"aug_feats={len(d['aug_names'])}")
    return datasets


