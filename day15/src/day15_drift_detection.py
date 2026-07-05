import os 
import warnings 
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd 
from scipy import stats 
from scipy.stats import ks_2samp

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR  = os.path.join(BASE_DIR, "..", "data", "processed")
OUT_DIR   = os.path.join(BASE_DIR, "output")

TICKERS        = ["SPY", "QQQ", "AAPL"]
REF_SIZE = 756 # ref window : 3 years
DRIFT_WIN_SIZE = 63 # curr window : 1 quarter
STEP_SIZE = 63 # check for the drift every 21 days
N_BINS = 10 # bins for popularity stability index

def compute_psi(reference: np.ndarray , current: np.ndarray , n_bins: int = N_BINS)-> float:
   eps = 1e-8
   breakpoints = np.percentile(reference , np.linspace(0,100 , n_bins+1))
   breakpoints[0] = -np.inf 
   breakpoints[-1] = np.inf 

   # Fraction in each bin under reference (P) and current (Q)
   ref_counts = np.histogram(reference , bins = breakpoints)[0]
   cur_counts = np.histogram(current , bins = breakpoints)[0]

   ref_pct = ref_counts / len(reference) + eps
   cur_pct = cur_counts / len(current) + eps

   # replacing zeros with epsilon to avoid the log(0) effect

   ref_pct = np.where(ref_pct == 0, eps , ref_pct)
   cur_pct = np.where(cur_pct == 0 , eps , cur_pct)

   psi = np.sum((cur_pct - ref_pct)* np.log(cur_pct / ref_pct))
   return float(psi)


def compute_jsd(reference: np.ndarray , current: np.ndarray , n_bins: int = 50)-> float:
   eps = 1e-10
   lo = min(reference.min() , current.min())
   hi = max(reference.max() , current.max())
   bins = np.linspace(lo , hi , n_bins + 1)

   p , _ = np.histogram(reference , bins = bins , density = True)
   q, _ = np.histogram(current , bins = bins , density = True)

   #Normalise as the density may not sum exactly 1 due to the bin width

   p = p / (p.sum() + eps)
   q = q / (q.sum() + eps)

   m = 0.5 * (p+q)

   # now clipping all the values which are there to avoid log(0)

   kl_pm = np.sum(p*np.log((p+eps) / (m+eps)))
   kl_qm = np.sum(q*np.log((q+eps) / (m+eps)))
   jsd = 0.5 * (kl_pm + kl_qm)
   return float(max(jsd , 0.0))
 
def rolling_drift_scan(series : pd.Series , ref_size : int = REF_SIZE , win_size: int = DRIFT_WIN_SIZE , step : int = STEP_SIZE )-> pd.DataFrame:
   ref_data = series.values[:ref_size]
   n = len(series)
   rows = []

   start = ref_size
   while start + win_size <= n:
    cur_data = series.values[start:start+win_size]
    date = series.index[start+win_size-1]
      
    psi = compute_psi(ref_data , cur_data)
    jsd = compute_jsd(ref_data , cur_data)

    ks_stat , ks_p = ks_2samp(ref_data , cur_data)

    # drift flag : any of the three methods triggers
    flag = int(psi>= 0.10 or ks_p < 0.05)

    rows.append({
        "date"     : date,
        "psi"      : round(psi,     4),
        "jsd"      : round(jsd,     4),
        "ks_stat"  : round(ks_stat, 4),
        "ks_pvalue": round(ks_p,    4),
        "drift_flag": flag,
        "psi_level": ("major" if psi >= 0.25 
                      else
                      "moderate" if psi >= 0.10 
                      else
                      "stable"),
    })
    start += step
   return pd.DataFrame(rows)

def feature_drift_scan( X: pd.DataFrame , ref_size : int = REF_SIZE , step : int = STEP_SIZE)-> pd.DataFrame:
   win_size = DRIFT_WIN_SIZE
   n = len(X)
   rows = []

   start = ref_size
   while start+win_size <=n:
      date = X.index[start + win_size - 1]
      row = {"date": date}
      for col in X.columns:
         ref = X[col].values[:ref_size]
         cur = X[col].values[start : start + win_size]
         row[f"psi_{col}"] = round(compute_psi(ref , cur) , 4)
      rows.append(row)
      start+=step
   return pd.DataFrame(rows)  

def run_drift_detection() -> dict:
   print(f"\n{'='*55}")
   print("  DAY 15 — Distribution Drift Detection")
   print(f"{'='*55}")
   print(f"  Reference window : {REF_SIZE} days")
   print(f"  Detection window : {DRIFT_WIN_SIZE} days")
   print(f"  Step size        : {STEP_SIZE} days")

   results = {}

   for ticker in TICKERS:
      path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
      if not os.path.exists(path):
         print(f"⚠ {path} not found")
         continue

      df = pd.read_csv(path, index_col="Date", parse_dates=True)
      rv_1d = (df["log_return"]**2 * 252).dropna()
      ret = df["log_return"].dropna()

      print(f"\n {ticker}: ")

      # 1 now drift on daily RV (primary target variable)
      rv_drift = rolling_drift_scan(rv_1d)
      n_drifts = rv_drift["drift_flag"].sum()
      psi_max = rv_drift["psi"].max() if not rv_drift.empty else 0.0
      print(f" RV Drift windows : {n_drifts}/{len(rv_drift)}")
      print(f"Max PSI: {psi_max:.4f} ({'major' if psi_max >= 0.25 else 'moderate' if psi_max >= 0.10 else 'stable'})")

      # 2 now drift on log returns (feature input)
      ret_drift = rolling_drift_scan(ret)

      # 3 Feature level PSI to identify which features drift
      from day15_online_har import build_har_xy
      X, y = build_har_xy(df)
      feat_drift = feature_drift_scan(X)

      results[ticker] = {
         "rv_drift": rv_drift,
         "ret_drift": ret_drift,
         "feat_drift": feat_drift,
      }

      # now saving these to the directories
      rv_drift.to_csv(os.path.join(OUT_DIR, f"day15_rv_drift_{ticker}.csv"), index=False)
      feat_drift.to_csv(os.path.join(OUT_DIR, f"day15_feature_drift_{ticker}.csv"), index=False)

      print(f"\n ✓ day15_rv_drift_{ticker}.csv")
      print(f"\n ✓ day15_feature_drift_{ticker}.csv")

   return results


if __name__ == "__main__":
    run_drift_detection()
    