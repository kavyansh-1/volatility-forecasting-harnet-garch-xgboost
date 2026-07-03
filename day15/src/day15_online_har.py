import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model  import Ridge
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS   = ["SPY", "QQQ", "AAPL"]
TEST_SIZE = 500 
BURN_IN = 252 # minimum no. of observations before making predictions 


# feature builder just as it was done earlier
def build_har_xy(df: pd.DataFrame)->tuple:
    rv_1d = df["log_return"]**2*252
    X = pd.DataFrame(index = df.index)
    X["rv_lag_1"] = rv_1d.shift(1)
    X["rv_lag_5"] = rv_1d.shift(1).rolling(5 , min_periods = 5).mean()
    X["rv_lag_21"] = rv_1d.shift(1).rolling(21 , min_periods = 21).mean()
    X["abs_ret_lag1"] = df["log_return"].abs().shift(1)

    y = rv_1d.shift(-1)
    combined = pd.concat([X , y.rename("target")], axis = 1).dropna()
    return combined.drop(columns=["target"]) , combined["target"]

# Using Method 1 The EWMA Sufficient Statistics Ridge
class EWMARidge:
    def __init__(self , n_features : int, alpha : float = 10.0 , lam : float = 0.97):
        self.n = n_features
        self.alpha = alpha
        self.lam = lam 

        self.S_XX = np.zeros((n_features , n_features))
        self.S_Xy = np.zeros(n_features)
        self.t = 0 
        self.w = np.zeros(n_features) # this is for the coeffecients
        self.b = 0.0 # this is the bias intercept and we keep it as 0

        # Now running the mean/std for online feature normalisation
        self.mu = np.zeros(n_features)
        self.var = np.ones(n_features)
        self.M2 = np.zeros(n_features)

    def _update_normalisation(self , x:np.ndarray)-> None:
        """ This is welford's online algo for running mean and the variance , it is
        numerically stable with no catastrophic cancellation"""

        self.t += 1
        delta = x - self.mu
        self.mu += delta / self.t
        delta2 = x - self.mu

        if self.t == 1:
            self.M2 = delta*delta2
        else:
            self.M2 = self.lam * self.M2 + (1-self.lam) * delta * delta2
        self.var = np.maximum(self.M2 , 1e-8)

    def _normalise(self , x: np.ndarray , y:float)-> np.ndarray:
        return (x - self.mu) / np.sqrt(self.var)
    
    def _update(self , x:np.ndarray , y:float)->None:
        self._update_normalisation(x)
        x_norm = self._normalise(x)
        self.S_XX = self.lam * self.S_XX + (1-self.lam) * np.outer(x_norm , x_norm)
        self.S_Xy = self.lam * self.S_Xy + (1 -self.lam) * x_norm * y 
        A = self.S_XX + self.alpha * np.eye(self.n)
        self.w = np.linalg.solve(A , self.S_Xy)

    def update(self , x:np.ndarray , y:float)-> None:
        self._update(x , y)
    
    def predict(self , x:np.ndarray)-> float:
        if self.t < 3:
            return float(self.S_Xy.mean()) if self.S_Xy.any() else 0.0
        x_norm = self._normalise(x)
        return float(np.dot(self.w , x_norm))
    
class SGDOnlineRidge:
    def __init__(self , n_features: int , alpha : float = 0.01 , lr: float = 0.01 , decay : float = 1e-4):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.alpha = alpha
        self.lr0 = lr
        self.decay = decay
        self.t = 0

        ## Running normalisation now 
        self.mu = np.zeros(n_features)
        self.var = np.ones(n_features)
        self.M2 = np.zeros(n_features)

    def _update_norm(self , x):
        self.t+=1
        delta = x - self.mu
        self.mu += delta / self.t
        delta2 = x-self.mu
        self.M2 = 0.99 * self.M2 + 0.01 * delta * delta2
        self.var = np.maximum(self.M2 , 1e-8)

    def _norm(self , x):
        return (x-self.mu) / np.sqrt(self.var)
    
    def update(self , x:np.ndarray , y: float)-> None:
        self._update_norm(x)
        x_n = self._norm(x)
        lr_t = self.lr0 / (1+self.decay * self.t)

        # Prediction error 
        err = y - (np.dot(self.w , x_n) + self.b)

        # Gradient step (MSE_loss + L2 regularisation) part 
        self.w = self.w + lr_t * (2 * err * x_n - 2 * self.alpha * self.w)
        self.b = self.b + lr_t * 2 * err

    def predict(self , x: np.ndarray)-> float:
        if self.t < 3:
            return 0.0
        x_n = self._norm(x)
        return float(np.dot(self.w , x_n) + self.b)
    
# Now offline Benchmark: Offline Ridge ( retrained every 21 days)
def offline_rolling_har(X:pd.DataFrame , y:pd.Series , retrain_freq : int = 21 , alpha : int = 10.0)->np.ndarray:
    n = len(X)
    preds = np.full(n , np.nan)

    for t in range(BURN_IN , n):
        if ( t - BURN_IN) % retrain_freq == 0:
            X_tr = X.iloc[:t]
            y_tr = y.iloc[:t]
            sc = StandardScaler()
            X_sc = sc.fit_transform(X_tr)
            ridge = Ridge ( alpha = alpha)
            ridge.fit(X_sc , y_tr)

        x_t = sc.transform(X.iloc[t:t+1])
        preds[t] = ridge.predict(x_t)[0]

    return preds 

# Now the main online comparison
def run_online_comparison()-> dict:
    print(f"\n{'='*55}")
    print("  DAY 15 — Online Learning Comparison")
    print(f"{'='*55}")
    print(f"BURN_IN = {BURN_IN} days (predictions start after)")

    results = {}
    all_rows = {}

    for ticker in TICKERS:
        path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
        if not os.path.exists(path):
            print(f"⚠ {path} not found")
            continue

        df = pd.read_csv(path , index_col = "Date" , parse_dates = True)
        X,y = build_har_xy(df)
        n = len(X)
        X_vals = X.values 
        y_vals = y.values

        print(f"\n {ticker} ({n} observations)")

        # Initialising the online models 
        n_feat = X_vals.shape[1]
        ewma = EWMARidge(n_feat, alpha = 10.0 , lam = 0.97)
        sgd = SGDOnlineRidge(n_feat , alpha = 0.01 , lr = 0.005 , decay = 1e-4)

        preds_ewma = np.full(n , np.nan)
        preds_sgd = np.full(n , np.nan)
        
        #Online loop: for each step t , Predict before updating 
        for t in range(n):
            x_t = X_vals[t]
            y_t = y_vals[t]

            if t>= BURN_IN:
                preds_ewma[t] = ewma.predict(x_t)
                preds_sgd[t] = sgd.predict(x_t)

            # Now updating with the true label ( that this is the online part)
            ewma.update(x_t, y_t)
            sgd.update(x_t , y_t)
        
        # Offline Baseline
        print("Running offline baseline (21 day - refitting)...")
        preds_offline = offline_rolling_har(X , y)
        ewma2 = EWMARidge(n_feat , alpha = 10.0 , lam = 0.97)
        coef_hist = []
        for t in range(n):
            if t >= BURN_IN and t%5==0:
                coef_hist.append({
                    "step" : t,
                    "date" : X.index[t],
                    **{f"w_{col}" : float(ewma2.w[i])
                    for i , col in enumerate(X.columns)}
                })
            ewma2.update(X_vals[t] , y_vals[t])
        coef_df = pd.DataFrame(coef_hist)

        results[ticker] = {
            "X" : X , 
            "y" : y, 
            "preds_ewma" : preds_ewma , 
            "preds_sgd" : preds_sgd,
            "preds_offline" : preds_offline, 
            "coef_df" : coef_df,

        }

        test_mask = np.zeros(n ,dtype=bool)
        test_mask[-TEST_SIZE:] = True
        valid = test_mask & ~np.isnan(preds_ewma)

        def rmse(a , b): return np.sqrt(np.mean((a-b)**2))
        
        for label, preds in [("EWMA_Online", preds_ewma),
                              ("SGD_Online",  preds_sgd),
                              ("Offline_21d", preds_offline)]:
            v = valid & ~np.isnan(preds)
            if v.sum() > 10:
                r = rmse(y_vals[v], preds[v])
                print(f"    {label:15s}: RMSE={r:.6f}")
                all_rows.append({
                    "Ticker": ticker, "Model": label,
                    "RMSE"  : round(r, 8), "N": int(v.sum()),
                })
    # Now finally saving the results 
    pd.DataFrame(all_rows).to_csv(os.path.join(OUT_DIR , "day15_online_metrics.csv") , index = False)
    # Now saving the coeffecient histories
    for ticker , res in results.items():
        res["coef_df"].to_csv(os.path.join(OUT_DIR , f"day15_coef_history_{ticker}.csv"), index = False)
    print(f"\n  ✓ day15_online_metrics.csv")
    print(f"  ✓ day15_coef_history_{{ticker}}.csv")
    return results


if __name__ == "__main__":
    run_online_comparison()




























