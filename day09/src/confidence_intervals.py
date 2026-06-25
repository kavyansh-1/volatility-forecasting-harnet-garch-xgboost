import os 
import warnings 
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd 
from sklearn.linear_model import Ridge 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR , ".." , "data" , "processed")
OUT_DIR = os.path.join(BASE_DIR , "output") 

TICKERS = ["SPY" , "QQQ" , "AAPL"]
N_BOOTSTRAP = 200 
CI_LEVEL = 0.90
BLOCK_SIZE = 22
TEST_SIZE = 500
ALPHA_HAR = 10.0

def block_bootstrap_residuals(residuals : np.ndarray , n_samples: int , block_size: int = 22 , seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    T = len(residuals)
    result = []

    while len(result) < n_samples:
        start = rng.integers(0, T - block_size + 1)
        block = residuals[start: start + block_size]
        result.extend(block.tolist())
    return np.array(result[:n_samples])

def compute_har_intervals(df : pd.DataFrame, ticker : str , n_boot : int = N_BOOTSTRAP , ci_level : float = CI_LEVEL ) -> pd.DataFrame:
    rv_1d = df["log_return"]**2*252
    feats = pd.DataFrame(index=df.index)
    feats["rv_lag_1"]  = rv_1d.shift(1)
    feats["rv_lag_5"]  = rv_1d.shift(1).rolling(5,  min_periods=5).mean()
    feats["rv_lag_21"] = rv_1d.shift(1).rolling(21, min_periods=21).mean()

    target   = rv_1d.shift(-1)
    combined = pd.concat([feats, target.rename("target")],
                          axis=1).dropna()

    X = combined.drop(columns=["target"])
    y = combined["target"] 
    n = len(X)

    split   = n - TEST_SIZE
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    pipe = Pipeline([
        ("sc",    StandardScaler()),
        ("ridge", Ridge(alpha=ALPHA_HAR)),
    ])
    pipe.fit(X_tr, y_tr)

    point_pred = pipe.predict(X_te)

    in_sample_pred = pipe.predict(X_tr)
    residuals      = y_tr.values - in_sample_pred

    print(f"  {ticker}: bootstrapping {n_boot} iterations "
          f"(block_size={BLOCK_SIZE})...")

    boot_preds = np.zeros((n_boot, len(X_te)))
    
    scaler     = StandardScaler()
    X_tr_sc    = scaler.fit_transform(X_tr)
    X_te_sc    = scaler.transform(X_te)

    for b in range(n_boot):
        e_star = block_bootstrap_residuals(
            residuals, n_samples=len(y_tr),
            block_size=BLOCK_SIZE, seed=b
        )
        y_star = in_sample_pred + e_star
        ridge_b = Ridge(alpha=ALPHA_HAR)
        ridge_b.fit(X_tr_sc, y_star)
        boot_preds[b, :] = ridge_b.predict(X_te_sc)

    alpha   = (1 - ci_level) / 2
    lower   = np.percentile(boot_preds, alpha * 100,       axis=0)
    upper   = np.percentile(boot_preds, (1 - alpha) * 100, axis=0)
    
    lower = np.maximum(lower, 0.0)

    actual    = y_te.values
    coverage  = np.mean((actual >= lower) & (actual <= upper)) * 100

    print(f"    Point RMSE  : {np.sqrt(np.mean((actual - point_pred)**2)):.6f}")
    print(f"    PI width    : {np.mean(upper - lower):.6f}")
    print(f"    Coverage    : {coverage:.1f}%  "
          f"(target={ci_level*100:.0f}%)")

    result = pd.DataFrame({
        "date"      : X_te.index,
        "ticker"    : ticker,
        "actual"    : actual,
        "point_pred": point_pred,
        "lower_90"  : lower,
        "upper_90"  : upper,
        "in_interval": ((actual >= lower) & (actual <= upper)).astype(int),
    })
    result["pi_width"] = result["upper_90"] - result["lower_90"]

    return result, float(coverage)

def run_confidence_intervals() -> pd.DataFrame:
    print(f"\n{'='*55}")
    print("  DAY 9 — Bootstrapped Prediction Intervals")
    print(f"{'='*55}")
    print(f"  Bootstrap iterations : {N_BOOTSTRAP}")
    print(f"  CI level             : {CI_LEVEL*100:.0f}%")
    print(f"  Block size           : {BLOCK_SIZE} days")

    all_dfs   = []
    cov_rows  = []

    for ticker in TICKERS:
        path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
        if not os.path.exists(path):
            print(f"\n Error {path} not found")
            continue

        df      = pd.read_csv(path, index_col="Date", parse_dates=True)
        res, cov = compute_har_intervals(df, ticker)
        all_dfs.append(res)
        cov_rows.append({
            "Ticker"        : ticker,
            "Coverage_pct"  : round(cov, 2),
            "Target_pct"    : CI_LEVEL * 100,
            "Mean_PI_width" : round(res["pi_width"].mean(), 6),
            "N"             : len(res),
        })

    combined = pd.concat(all_dfs, ignore_index=True)
    cov_df   = pd.DataFrame(cov_rows)

    out1 = os.path.join(OUT_DIR, "day09_prediction_intervals.csv")
    out2 = os.path.join(OUT_DIR, "day09_pi_coverage.csv")
    combined.to_csv(out1, index=False)
    cov_df.to_csv(out2,   index=False)

    print(f"\n Prediction intervals done → {out1}")
    print(f" Coverage summary done    → {out2}")
    print(f"\n Coverage summary:")
    print(cov_df.to_string(index=False))

    return combined, cov_df
