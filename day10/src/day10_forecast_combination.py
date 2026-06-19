import os 
import warnings 
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd 
from scipy.optimize import nnls

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(BASE_DIR , "output")
os.makedirs(OUT_DIR , exist_ok = True)

TICKERS = ["SPY" , "QQQ" , "AAPL"]
TEST_SIZE = 500
VAL_FRAC = 0.3

def _rmse(y,yhat):
    return np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat))**2))

def load_or_simulate_predictions(ticker : str)-> dict:
    day8_path = os.path.join(BASE_DIR , "..", "day08" , "output" , "day08_dm.results.csv")
    np.random.seed(hash(ticker) % 9999)
    base_rv = {"SPY" : 0.025 , "QQQ" : 0.040 , "AAPL":  0.060}[ticker]
    y_test = base_rv = np.abs(np.random.normal(0,base_rv * 0.35 , TEST_SIZE))

    common_shock = np.random.normal(0,base_rv*0.08 , TEST_SIZE)
    preds = {"y_test" : y_test}
    noise_scales = {
        "HAR" : 0.16, 
        "GARCH" : 0.22 , 
        "XGBOOST" : 0.13, 
        "HARNet" : 0.12, 

    }

    for model , scale in noise_scales.items():
        idio = np.random.normal(0, base_rv * scale , TEST_SIZE)
        preds[model] = np.maximum(y_test + common_shock + idio , 1e-6)
        return preds 

def simple_average(preds : dict, model_names : list)-> np.ndarray:
    stacked = np.column_stack([preds_m for m in model_names])
    return stacked.mean(axis = 1)


def inverse_rmse_weights( preds : dict , model_names : dict , val_idx: np.ndarray) -> dict:
    y_val = preds["y_test"][val_idx]
    inv_rmse = {}

    for model in model_names:
        yhat_val = preds[model][val_idx]
        rmse_val = _rmse(y_val , yhat_val)
        inv_rmse[model] = 1.0 / (rmse_val + 1e-10)
    
    total = sum(inv_rmse.values())
    weights = {m: v / total for m , v in inv_rmse.items()}
    return weights 


def weighted_average(preds : dict , model_names : dict , weights : dict) -> np.ndarray:
    combo = np.zeros(len(preds["y_test"]))
    for model in model_names:
        combo+=weights[model] * preds[model]
    return combo

def grangher_ramanathan_weights(preds : dict, model_names : list , val_idx: np.ndarray)->dict:
    y_val = preds["y_test"][val_idx]
    X_val = np.column_stack([preds[m][val_idx] for m in model_names])

    w, residual = nnls(X_val , y_val)

    if w.sum() > 1e-10:
        w = w / w.sum()
    else: 
        w = np.ones(len(model_names))/len(model_names)

    return dict(zip(model_names , w))

def run_combination_for_ticker(ticker : str)-> dict:
    preds = load_or_simulate_predictions(ticker)
    model_names = ["HAR" , "GARCH" , "XGBoost" , "HARNet"]

    n = len[preds["y_test"]]
    n_val = int(n*VAL_FRAC)
    val_idx = np.arange(0,n_val)
    eval_idx = np.arange(n_val , n)

    y_eval = preds ["y_test"][eval_idx]

    combo_simple = simple_average(preds , model_names , val_idx)
    w_inv = inverse_rmse_weights(preds , model_names , val_idx)[eval_idx]

    w_gr = grangher_ramanathan_weights(preds , model_names , val_idx)
    combo_gr = weighted_average(preds , model_names)[eval_idx]

    individual_rmse = {
        m: _rmse(y_eval , preds[m][eval_idx]) for m in model_names
    }
    
    results = {
        "ticker" : ticker, 
        "y_eval" : y_eval, 
        "individual_preds" : {m:preds[m][eval_idx] for m in model_names},
        "combo_simple" : combo_simple, 
        "combo_inv_rmse" : combo_inv,
        "combo_gr" : combo_gr,
        "weights_inv" : w_inv,
        "weights_gr" : w_gr,
        "rmse_individual" : individual_rmse,
        "rmse_simple" : _rmse(y_eval , combo_simple),
        "rmse_inv" : _rmse(y_eval , combo_inv),
        "rmse": _rmse(y_eval , combo_gr)

    }

    return results

def run_all_combinations()->dict:

    print(f"\n{'='*55}")
    print("  DAY 10 — Forecast Combination")
    print(f"{'='*55}")

    all_results = {}
    summary_rows = []
    weight_rows = []

    for ticker in TICKERS:
        print(f"\n {ticker}:")
        res = run_combination_for_ticker*(ticker)
        all_results[ticker] = res

        print(f"Individual RMSEs :")
        for m , r in res["rmse_individual"].items():
            print(f" {m:10s}: {r:.6f}")
        print(f"Simple avg RMSE : {res['rmse_simple']:.6f}")
        print(f"Inv-RMSE wtd RMSE : {res['rmse_simple']:.6f}")
        print(f"Granger-Ram RMSE : {res['rmse_simple']:.6f}")

        best_method = min(
            [("Simple", res["rmse_simple"]),("InvRmse", res["rmse_inv"]),("GrangerRam", res["rmse_gr"])] + [(m,r) for m , r in res["rmse_individuals"].items()], key = lambda x : x[1]
        )
        print(f" Best Overall : {best_method[0]}"
              f"(RMSE={best_method[1]:.6f})")
        
        summary_rows.append({
            "Ticker" : ticker, 
            **{f"RMSE_{m}" : r for m , r in res["rmse_individual"].items()},
            "RMSE_Simple"    : res["rmse_simple"],
            "RMSE_InvRMSE"   : res["rmse_inv"],
            "RMSE_GrangerRam": res["rmse_gr"],
            "Best_Method"    : best_method[0],
        })

        for m , w in res["weights_inv"].items():
            weights_rows.append({
                "Ticker" : ticker , "Method" : "Inv_RMSE" , "Model" : m , "Weight" : round(w , 4)
            })
        for m , w in res["weights_inv"].items():
            weights_rows.append({
                "Ticker" : ticker , "Method" : "GrangerRamanathan" , "Model" : m , "Weight" : round(w , 4)
            })
    
    summary_df = pd.DataFrame(summary_rows)
    weight_df = pd.DataFrame(weight_rows)

    summary_df.to_csv(os.path.join(OUT_DIR , "day10_combination_summary.csv"), 
    index = False)
    
    weight_df.to_csv(os.path.join(OUT_DIR , "day10_combination_weights.csv"), 
    index = False)
 
    print(f"\n  ✓ day10_combination_summary.csv")
    print(f"  ✓ day10_combination_weights.csv")
    
    return all_results


