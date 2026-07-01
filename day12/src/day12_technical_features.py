import os 
import numpy as np 
import pandas as pd 

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR , ".." , "data" , "processed")
OUT_DIR = os.path.join(BASE_DIR , "output")
os.makedirs(OUT_DIR , exist_ok = True)

TICKERS = ["SPY" , "QQQ" , "AAPL"]

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    h = df["High"]
    lo = df["Low"]

    c_prev = c.shift(1)
    tr = pd.concat([(h-lo) , (h-c_prev).abs() ,(lo-c_prev).abs()] , axis = 1).max(axis = 1)

    for w in [5,14,21]:
        df[f"atr_{w}d"] = tr.rolling(w , min_periods = w).mean().shift(1)
        df[f"atr_norm_{w}d"] = df[f"atr_{w}d"] / (c.shift(1) + 1e-6)

    for w in [20]:
        ma = c.rolling(w , min_periods = w).mean()
        std = c.rolling(w , min_periods = w).std()
        df[f"bb_width_{w}d"] = (2*std/(ma+1e-6)).shift(1)
        df[f"bb_pct_b_{w}d"] = ((c - (ma - 2 * std)) / (4*std+1e-6)).shift(1)

    delta = c.diff()
    gain = delta.clip(lower = 0).rolling(14 , min_periods = 14).mean()
    loss = (-delta.clip(upper = 0)).rolling(14 , min_periods = 14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - 100/(1+rs)
    df["rsi_14d"] = rsi.shift(1)
    df["rsi_extreme_dist"] = (rsi-50).abs().shift(1)

    for w in [10,50]:
        ma = c.rolling(w , min_periods = w).mean()
        df[f"price_ma_gap_{w}d"] = ((c-ma) / (ma + 1e-6)).shift(1)
        df[f"price_ma_gap_abs_{w}d"] = df[f"price_ma_gap_{w}d"].abs()

    ema12 = c.ewm(span=12 , min_periods = 12).mean()
    ema26 = c.ewm(span = 26 , min_periods = 26).mean()
    macd = ema12 - ema26
    df["macd_norm"] = (macd / (c + 1e-6)).shift(1)
    df["macd_abs_norm"] = df["macd_norm"].abs()

    df["macd_delta"] = (macd - macd.shift(1)).shift(1) / (c.shift(1) + 1e-6)

    if "Volume" in df.columns:
        vol = df["Volume"].astype(float)
        vol_ma = vol.rolling(21 , min_periods = 10).mean()
        df["vol_ratio_21d"] = (vol / (vol_ma + 1e-6)).shift(1) 
        df["vol_x_ret_lag_1"] = (df["vol_ratio_21d"] * df["log_return"].abs().shift(1))

    sign_ret = np.sign(df["log_return"])
    streak = pd.Series(0, index=df.index)
    for i in range(1 , len(sign_ret)):
        if sign_ret.iloc[i] == sign_ret.iloc[i-1] and sign_ret.iloc[i]!=0:
            streak.iloc[i] = streak.iloc[i-1]
        else:
            streak.iloc[i] = 0
    df["return_streak"] = streak.shift(1)

    return df 

def run_technical_features(vol_dfs: dict = None) -> dict:

    print(f"\n{'-'*50}")
    print("Technical Indicator Features")
    print('-' * 50)

    results = {}
    for ticker in TICKERS:
        if vol_dfs and ticker in vol_dfs:
            df = vol_dfs[ticker]
        else:
            path = os.path.join(DATA_DIR , f"{ticker}_processed.csv")
            if not os.path.exists(path):
                print(f" Oops , {path} not found.. ")
                continue
            df = pd.read_csv(path , index_col = "Date" , parse_dates = True)
        
        df = add_technical_features(df) 

        tech_cols = [c for c in df.columns if any(k in c for k in ["atr" , "bb_" , "rsi" , "macd" , "vol_ratio" , "vol_x_ret" , "price_ma" , "return_streak"])]

        print(f" {ticker} : {len(tech_cols)} technical features added yayayayayya")
        results[ticker] = df

    return results













