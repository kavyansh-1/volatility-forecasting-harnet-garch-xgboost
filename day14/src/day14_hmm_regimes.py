import os 
import warnings 
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd 
from hmmlearn import hmm 

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR , ".." , "data" , "processed")
OUT_DIR = os.path.join(BASE_DIR , "output")
os.makedirs(OUT_DIR , exist_ok = True)

TICKERS = ["SPY" , "QQQ" , "AAPL"]
N_STATES = 3
TEST_SIZE = 500

def prepare_hmm_input(df : pd.DataFrame)-> np.ndarray:
    rv = df["rv_rolling_21d"].dropna()
    log_rv = np.log(rv.values + 1e-10).reshape(-1,1)
    return log_rv , rv.index

def fit_hmm(log_rv: np.ndarray , n_states: int = N_STATES , n_iter : int = 200, seed: int = 42)-> hmm.GaussianHMM:
    model = hmm.GaussianHMM( 
        n_components = n_states , 
        covariance_type = "diag" , 
        n_iter = n_iter, 
        random_state = seed, 
        verbose = False,
        )
    model.fit(log_rv)
    return model

def decode_regimes(model: hmm.GaussianHMM , log_rv : np.ndarray)-> np.ndarray:
    _ , state_sequence = model.decode(log_rv , algorithm = "viterbi")
    return state_sequence

def label_regimes_by_vol(model: hmm.GaussianHMM , state_sequence : np.ndarray)-> np.ndarray:
    means = model.means_.flatten()
    order = np.argsort(means)
    remap = {old : new for new  , old in enumerate (order)}
    return np.array([remap[s] for s in state_sequence])

def extract_regime_statistics(model : hmm.GaussianHMM , labels: np.ndarray , log_rv : np.ndarray)-> pd.DataFrame:
    means = model.means_.flatten()
    stds = np.sqrt(model.covars_.flatten())
    transmat = model.transmat_


    rows = []
    regime_names = ["Low" , "Medium" , "High"]
    order = np.argsort(means)

    for rank , state_idx in enumerate(order):
        raw_mean = np.exp(means[state_idx])
        raw_stds = np.exp(stds[state_idx])
        freq = (labels == rank).mean()*100
        persist = transmat[state_idx , state_idx]

        rows.append({
            "Regime" : regime_names[rank], 
            "Mean_RV_Ann" : round(raw_mean , 4),
            "Std_RV_Ann" : round(raw_stds , 4), 
            "Frequency_pct" : round(freq , 2),
            "Persistence" : round(persist , 4),
        })

    return pd.DataFrame(rows)

def compute_regime_posteriors(model: hmm.GaussianHMM , log_rv : np.ndarray)->np.ndarray:
    posteriors = model.predict_proba(log_rv)
    return posteriors

def run_hmm_for_ticker(df: pd.DataFrame , ticker : str)-> dict:
    print(f"\n  {ticker}:")
    log_rv , dates = prepare_hmm_input(df)
    model = fit_hmm(log_rv)
    raw_states = decode_regimes(model , log_rv)
    regime_labels = label_regimes_by_vol(model , raw_states)
    posteriors = compute_regime_posteriors(model , log_rv)
    stats_df = extract_regime_statistics(model , regime_labels , log_rv)

    print(stats_df.to_string(index = False))
    print(f"\n Transition Matrix:")
    order = np.argsort(model.means_.flatten())
    print(pd.DataFrame(
        model.transmat_[np.ix_(order, order)],
        index = ["Low→", "Med→", "High→"],
        columns = ["Low→", "Med→", "High→"],
    ).round(3).to_string())

    rv_index = df["rv_rolling_21d"].dropna().index
    regime_series = pd.Series(regime_labels , index = rv_index , name = "regime")
    post_df = pd.DataFrame(posteriors , index = rv_index, columns = ["post_low" , "post_med" , "post_high"])
    

    return{"ticker" : ticker , 
           "model" : model , 
           "regime_series" : regime_series , 
           "posteriors" : post_df , 
           "stats_df" : stats_df, 
           "dates" : dates, 
           "log_rv" : log_rv, 
           "order" : order,}

def run_hmm_all_tickers()->dict:
    print(f"\n{'='*55}")
    print("  DAY 14 — HMM Regime Detection")
    print(f"{'='*55}")
    print(f" N_STATES = {N_STATES} Algorithm =Viterbi+ForwardBackward")

    results = {}
    all_regime_rows = []

    for ticker in TICKERS:
        path = os.path.join(DATA_DIR , f"{ticker}_processed.csv")
        if not os.path.exists(path):
            print(f" {path} not found")
            continue
        df = pd.read_csv(path , index_col = "Date" , parse_dates = True)
        res = run_hmm_for_ticker(df , ticker)
        results[ticker] = res 

        reg = res["regime_series"].reset_index()
        reg.columns = ["date" , "regime"]
        reg["ticker"] = ticker

        post = res["posteriors"].reset_index()
        post.columns = ["date" , "post_low" , "post_med" , "post_high"]
        combined = reg.merge(post , on = "date")
        all_regime_rows.append(combined)

    if all_regime_rows: 
        regime_df = pd.concat(all_regime_rows , ignore_index = True)
        out = os.path.join(OUT_DIR , "day14_hmm_regimes.csv")
        regime_df.to_csv(out , index = False)
        print(f"\n  ✓ day14_hmm_regimes.csv  ({len(regime_df)} rows)")
    
    
    return results





    



    


