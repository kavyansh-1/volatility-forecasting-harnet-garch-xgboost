import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.feature_engineering import load_ticker_frames, prepare_xy, train_test_split_ts

warnings.filterwarnings("ignore")

XGB_PARAM_DIST = {
    "n_estimators": [50, 100, 200, 300, 500],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.005, 0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "reg_alpha": [0.0, 0.01, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    "gamma": [0.0, 0.01, 0.1, 0.5],
}

def tune_xgboost(
    df: pd.DataFrame,
    ticker: str,
    n_iter: int = 30,
    cv_splits: int = 5,
    seed: int = 42,
    test_size: int = 500,
) -> dict:
    
    print(f"Tuning XGBoost for {ticker} ({n_iter} random configs x {cv_splits} folds)...")
    X, y, rv_1d = prepare_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split_ts(X, y, test_size=test_size)

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    base_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )

    rs = RandomizedSearchCV(
        base_model,
        param_distributions=XGB_PARAM_DIST,
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        refit=True,
        random_state= seed,
        n_jobs=-1,
        error_score=np.nan,
    )

    rs.fit(X_tr, y_tr)

    best_params = rs.best_params_
    best_cv_rmse = -rs.best_score_

    importance = pd.Series(
        rs.best_estimator_.feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    
    print(f"Best CV-RMSE: {best_cv_rmse:.6f}")
    print(f"Top 3 Features: {list(importance.index[:3])}")

    cv_results = pd.DataFrame(rs.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score").head(10)
    cv_results["mean_rmse"] = -cv_results["mean_test_score"]

    return {
        "ticker": ticker,
        "best_params": best_params,
        "cv_rmse": round(best_cv_rmse, 8),
        "model": rs.best_estimator_,
        "importance": importance,
        "cv_results": cv_results,
        "X_train": X_tr,
        "X_test": X_te,
        "y_train": y_tr,
        "y_test": y_te,
        "rv_1d": rv_1d,
    }

def tune_xgboost_all_tickers(dfs: dict, **kwargs) -> dict:
    print("\n---- XGBoost Model Tuning ----")
    results = {}
    for ticker, df in dfs.items():
        results[ticker] = tune_xgboost(df, ticker, **kwargs)
    return results


if __name__ == "__main__":
    tickers = ["SPY", "QQQ", "AAPL"]
    dfs = load_ticker_frames(tickers)
    results = tune_xgboost_all_tickers(dfs, n_iter=10)
    for t, r in results.items():
        print(f"{t}: CV-RMSE={r['cv_rmse']:.6f}, best depth={r['best_params'].get('max_depth')}")
