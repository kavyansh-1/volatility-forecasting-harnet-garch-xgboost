import os 
import warnings 
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd 
from arch import arch_model

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS   = ["SPY", "QQQ", "AAPL"]
TEST_SIZE = 500

def fit_gjr_garch(returns : pd.Series , train_emd : int)->dict:
   r_train = returns.iloc[:train_end] * 100

   model = arch_model( 
       r_train , 
       vol = "Garch" , 
       p = 1 , o = 1 , q = 1, 
       dist = "t", 
       mean = "AR", 
       lags = 1,
       ) 
   try:
      result = model.fit(disp = "off" , show_warning = False)
   except Exception as e:
      print(f"GJR - GARCH fit failed: {e}")
      return None
   
   #Extract key parameters 
   params = result.params
   gamma = params.get("gamma[1]" , params.get("gamma" , np.nan))
   alpha = params.get("alpha[1]" , params.get("alpha" , np.nan))
   beta = params.get("beta[1]" , params.get("beta" , np.nan))
   omega = params.get("omega[1]" , np.nan)

   # 1-step-ahead forecasts on test set (rolling)
   test_forecasts = []
   r_all = returns * 100
   for t in range(train_end , len(returns)):
      hist = r_all.iloc[:t]
      try: 
         m = arch_model(hist, vol = "Garch" , p = 1 , o = 1 , q = 1 , dist = "t" , mean = "AR" , lags = 1)
         res = m.fit(disp="off" , show_warning = False , starting_values  = result.params.values)
         fc = res.forecast(horizon = 1 , reindex = False)
         h_next = fc.variance.values[0,0] / (100**2)*252
      except Exception:
         h_next = np.nan
      test_forecasts.append(h_next)
   
   return{
      "model" : "GJR-GARCH", 
      "result" : result, 
      "omega" : omega, 
      "alpha" : alpha, 
      "gamma" : gamma, 
      "beta" : beta, 
      "persistence" : alpha + beta + 0.5 * gamma,
      "aic" : result.aic, 
      "bic" : result.bic, 
      "test_forecasts" : np.array(test_forecasts),
   }

def fit_egarch(returns : pd.Series , train_end : int)-> dict:
   r_train = returns.iloc["train_end"] * 100

   model = arch_model(
      r_train , vol = "EGARCH" , p = 1 , q = 1, dist = "t" , mean = "AR" , lags = 1,
   )

   try: 
      result = model.fit(disp = "off" , show_warning = False)
   except Exception as e:
      print(f"EGARCH Fit failed : {e}")
      return None
   
   params = result.params
   gamma = params.get("gamma[1]" , params.get("gamma" , np.nan))
   alpha = params.get("alpha[1]", params.get("alpha" , np.nan))
   beta = params.get("beta[1]", params.get("beta" , np.nan))
   omega = params.get("omega" , np.nan)

   test_forecasts = []
   r_all = returns * 100
   for t in range( train_end , len(returns)):
      hist = r_all.iloc[:t]
      try: 
         m = arch_model ( hist , vol = "EGARCH" , p = 1 , q = 1 , dist = "t" , mean = "AR" , lags = 1)
         res = m.fit(disp = "off" , show_warning = False , starting_values = result.params.values)
         fc = res.forecast(horizon = 1 , reindex = False)
         h_next = fc.variance.values[0,0] / (100**2)*252
      except Exception:
         h_next = np.nan
      test_forecasts.append(h_next)

   
   return {
        "model"         : "EGARCH",
        "result"        : result,
        "omega"         : omega,
        "alpha"         : alpha,
        "gamma"         : gamma,
        "beta"          : beta,
        "aic"           : result.aic,
        "bic"           : result.bic,
        "test_forecasts": np.array(test_forecasts),
    }   

def news_impact_curve(gjr : dict = None , egarch : dict = None , n_points: int = 100)-> pd.DataFrame:
   if gjr is not None and gjr.get("result") is not None:
      res = gjr["result"]
      params = res.params
      omega = params.get("omega" , 0)
      alpha = params.get("alpha[1]" , params.get("alpha" , 0.05))
      beta = params.get("beta[1]" , params.get("beta" , 0.90))
      gamma_p = params.get("gamma[1]" , params.get("gamma",0.05))

      #Unconditional variance under GJR-GARCH
      h_bar = omega / (1-alpha-beta-0.5*gamma_p + 1e-10)
      h_bar = max(h_bar , 1e-4)

      for eps in eps_grid:
         I_neg = 1.0 if eps < 0 else 0.0
         h_next = omega + (alpha + gamma_p * I_neg)*eps**2 + beta * h_bar
         rows.append({"eps" : eps , "h_next": h_next , "model" : "GJR-GARCH"})

   
   if egarch is not None and egarch.get("result") is not None:
        res    = egarch["result"]
        params = res.params
        omega  = params.get("omega", -0.1)
        alpha  = params.get("alpha[1]", params.get("alpha", 0.10))
        gamma_p= params.get("gamma[1]", params.get("gamma", -0.05))
        beta   = params.get("beta[1]",  params.get("beta",  0.97))

   #Unconditional Log variance for EGARCH : omega / (1-beta)
   log_h_bar = omega / (1-abs(beta) + 1e-10)
   h_bar = np.exp(log_h_bar)
   h_bar = max(h_bar , 1e-6)
   E_abs_z = np.sqrt(2 / np.pi) #E[|Z|] for Z~N(0,1)

   for eps in eps_grid:
      z = eps / np.sqrt(h_bar)
      log_h = (omega + alpha * (abs(z) - E_abs_z) + gamma_p * z + beta * log_h_bar)
      h_next = np.exp(log_h)
      rows.append({"eps":eps , "h_next" : h_next , "model" : "EGARCH"})

   return pd.DataFrame(rows)

def run_asymmetric_garch() -> dict:
    """Fit GJR-GARCH and EGARCH for all tickers."""
    print(f"\n{'='*55}")
    print("DAY 17 — Asymmetric GARCH Models")
    print(f"{'='*55}")

    results = {}
    met_rows = []

    for ticker in TICKERS:
        path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
        if not os.path.exists(path):
            continue
   
        df = pd.read_csv(path , index_col = "Date" , parse_dates = True)
        ret = df["log_return"].dropna()
        n = len(ret)
        train_end = n - TEST_SIZE

        print(f"\n {ticker} (train={train_end} , test={TEST_SIZE}):")
        print("Fitting GJR-GARCH...")
        gjr_res = fit_gjr_garch(ret , train_end)
        print("Fitting EGARCH...")
        eg_res = fit_egarch(ret, train_end)
        nic_df = news_impact_curve(gjr_res , eg_res)

        
        if gjr_res:
            print(f"    GJR: alpha={gjr_res['alpha']:.4f}"
                  f"  gamma={gjr_res['gamma']:.4f}"
                  f"  persist={gjr_res['persistence']:.4f}")

        if eg_res:
            print(f"    EG:  alpha={eg_res['alpha']:.4f}"
                  f"  gamma={eg_res['gamma']:.4f}"
                  f"  beta={eg_res['beta']:.4f}") 
      
        results[ticker] = {
           "ret" : ret, 
           "gjr" : gjr_res, 
           "egarch" : eg_res, 
           "nic_df" : nic_df, 
           "dates_test" : ret.index[-TEST_SIZE:], 
           "y_test" : (ret.values[-TEST_SIZE:]**2*252), 



        }

        for model_dict in [gjr_res , eg_res]:
           if model_dict is None:
              continue 
           preds = model_dict["test_forecasts"]
           valid = ~np.isnan(preds)
           if valid.sum() < 10:
              continue
           y_v = results[ticker]["y_test"]["len(preds)"][valid]
           p_v = preds[valid]
           rmse = np.sqrt(np.mean((y_v-p_v)**2))
           qlike = np.mean(np.log(np.maximum(p_v, 1e-8)) + np.maximum(y_v, 1e-8)/np.maximum(p_v , 1e-8))
           met_rows.append{
            "Ticker": ticker, 
            "Model" : model_dict["model"], 
            "RMSE" : round(rmse , 8), 
            "QLIKE" : round(qlike , 6), 
            "AIC" : round(model_dict["aic"] , 2), 
            "BIC" : round(model_dict["bic"],2),
            }
   
    met_df = pd.DataFrame(met_rows)
    met_df.to_csv(os.path.join(OUT_DIR, "day17_garch_metrics.csv"),
                  index=False)
    print(f"\n  ✓ day17_garch_metrics.csv")
    print(met_df.to_string(index=False))
    return results  

           






