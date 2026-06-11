import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.feature_engineering import load_ticker_frames


def build_har_features(df: pd.DataFrame, include_gk: bool = True) -> tuple:
   rv_1d = df["log_return"] ** 2 * 252
   target = rv_1d.shift(-1)

   feats = pd.DataFrame(index=df.index)
   feats["rv_lag_1"] = rv_1d.shift(1)
   feats["rv_lag_5"] = rv_1d.shift(1).rolling(5, min_periods=5).mean()
   feats["rv_lag_21"] = rv_1d.shift(1).rolling(21, min_periods=21).mean()

   if include_gk and "gk_vol_21d" in df.columns:
      feats["gk_vol_21d"] = df["gk_vol_21d"].shift(1)

   combined = pd.concat([feats, target.rename("target")], axis=1).dropna()
   X = combined.drop(columns=["target"])
   y = combined["target"]
   return X, y


def tune_har(
   df: pd.DataFrame,
   ticker: str,
   alphas: list | None = None,
   cv_splits: int = 5,
) -> dict:
   if alphas is None:
      alphas = [0.0001, 0.001, 0.01, 1.0, 10.0, 100.0]

   tscv = TimeSeriesSplit(n_splits=cv_splits)
   best_score = np.inf
   best_cfg = None
   best_pipe = None
   best_Xtr = None
   best_Xte = None
   best_ytr = None
   best_yte = None

   for include_gk in [True, False]:
      X, y = build_har_features(df, include_gk=include_gk)

      n = len(X)
      tr_end = max(1, n - 500)
      X_tr, X_te = X.iloc[:tr_end], X.iloc[tr_end:]
      y_tr, y_te = y.iloc[:tr_end], y.iloc[tr_end:]
      if len(X_tr) < max(2, cv_splits + 1) or len(X_te) == 0:
         continue

      pipe = Pipeline([
         ("scaler", StandardScaler()),
         ("ridge", Ridge()),
      ])

      param_grid = {"ridge__alpha": alphas}
      gs = GridSearchCV(
         pipe,
         param_grid,
         cv=tscv,
         scoring="neg_root_mean_squared_error",
         refit=True,
         n_jobs=-1,
      )
      gs.fit(X_tr, y_tr)

      cv_rmse = -gs.best_score_
      if cv_rmse < best_score:
         best_score = cv_rmse
         best_cfg = {
            "ticker": ticker,
            "best_alpha": gs.best_params_["ridge__alpha"],
            "include_gk": include_gk,
            "cv_rmse": round(cv_rmse, 8),
            "n_features": X.shape[1],
         }
         best_pipe = gs.best_estimator_
         best_Xtr = X_tr
         best_Xte = X_te
         best_ytr = y_tr
         best_yte = y_te

   if best_cfg is None:
      return {
         "ticker": ticker,
         "error": "insufficient data to tune HAR model",
         "pipeline": None,
         "X_train": None,
         "y_train": None,
         "X_test": None,
         "y_test": None,
      }

   print(
      f"{ticker} HAR best: alpha_best={best_cfg['best_alpha']}, "
      f"GK={best_cfg['include_gk']}, cv-RMSE={best_cfg['cv_rmse']:.6f}"
   )

   return {
      **best_cfg,
      "pipeline": best_pipe,
      "X_train": best_Xtr,
      "y_train": best_ytr,
      "X_test": best_Xte,
      "y_test": best_yte,
   }


def tune_har_all_tickers(dfs: dict, **kwargs) -> dict:
   print("\n--- HAR MODEL TUNING ---")
   results = {}
   for ticker, df in dfs.items():
      results[ticker] = tune_har(df, ticker, **kwargs)
   return results


if __name__ == "__main__":
   import warnings

   warnings.filterwarnings("ignore")

   tickers = ["SPY", "QQQ", "AAPL"]
   dfs = load_ticker_frames(tickers)
   results = tune_har_all_tickers(dfs)
   for ticker, result in results.items():
      if "best_alpha" in result:
         print(
            f"{ticker}: alpha={result['best_alpha']} , "
            f"Gk={result['include_gk']}, RMSE={result['cv_rmse']:.6f}"
         )
      else:
         print(f"{ticker}: {result['error']}")






