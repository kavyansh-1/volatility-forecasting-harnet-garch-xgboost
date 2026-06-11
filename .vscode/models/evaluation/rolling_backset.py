import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from models.feature_engineering import prepare_xy
from models.tune_har_model import build_har_features

warnings.filterwarnings("ignore")

try:
    from arch import arch_model

    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    eps = 1e-10
    y_true = np.maximum(y_true, eps)
    y_pred = np.maximum(y_pred, eps)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps)) * 100
    qlike = np.mean(np.log(y_pred) + y_true / y_pred)

    if len(y_true) > 1 and len(y_pred) > 1:
        delta_true = np.diff(y_true)
        delta_pred = np.diff(y_pred)
        diracc = np.mean(np.sign(delta_true) == np.sign(delta_pred)) * 100
    else:
        diracc = np.nan

    return {
        "Model": label,
        "RMSE": round(rmse, 8),
        "MAE": round(mae, 8),
        "MAPE": round(mape, 4),
        "QLIKE": round(qlike, 6),
        "DirAcc": round(diracc, 2) if not np.isnan(diracc) else np.nan,
        "N": len(y_true),
    }


def backtest_har(
    df: pd.DataFrame,
    ticker: str,
    best_alpha: float = 0.01,
    include_gk: bool = True,
    initial_train_size: int = 756,
    step_size: int = 21,
) -> pd.DataFrame:
    X, y = build_har_features(df, include_gk=include_gk)
    n = len(X)

    forecasts = []
    actuals = []
    dates = []

    for start in range(initial_train_size, n, step_size):
        end = min(start + step_size, n)
        X_tr = X.iloc[:start]
        y_tr = y.iloc[:start]
        X_te = X.iloc[start:end]
        y_te = y.iloc[start:end]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=best_alpha)),
        ])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)

        forecasts.extend(preds.tolist())
        actuals.extend(y_te.tolist())
        dates.extend(X_te.index.tolist())

    return pd.DataFrame({
        "Date": dates,
        "Actual": actuals,
        "HAR_Pred": forecasts,
    }).set_index("Date")


def backtest_garch(
    df: pd.DataFrame,
    ticker: str,
    best_p: int = 1,
    best_q: int = 1,
    best_vol_model: str = "GARCH",
    best_dist: str = "t",
    initial_train_size: int = 756,
    step_size: int = 21,
) -> pd.DataFrame:
    if not ARCH_AVAILABLE:
        print("arch not available, skipping GARCH backtest")
        return pd.DataFrame()

    returns = df["log_return"].dropna()
    rv_1d = returns**2 * 252
    n = len(returns)

    forecasts = []
    actuals = []
    dates = []

    for start in range(initial_train_size, n, step_size):
        end = min(start + step_size, n)
        r_tr = returns.iloc[:start] * 100
        rv_te = rv_1d.iloc[start:end]

        try:
            am = arch_model(
                r_tr,
                p=best_p,
                q=best_q,
                vol=best_vol_model,
                dist=best_dist,
                rescale=False,
            )
            result = am.fit(disp="off", show_warning=False)
            fc = result.forecast(horizon=1, reindex=False)
            h = fc.variance.values.flatten()
            h_scaled = (h / 10000.0) * 252

            n_te = len(rv_te)
            h_use = np.full(n_te, h_scaled[-1] if len(h_scaled) else np.nan)

            forecasts.extend(h_use.tolist())
            actuals.extend(rv_te.tolist())
            dates.extend(rv_te.index.tolist())
        except Exception:
            n_te = len(rv_te)
            forecasts.extend([np.nan] * n_te)
            actuals.extend(rv_te.tolist())
            dates.extend(rv_te.index.tolist())

    return pd.DataFrame({
        "Date": dates,
        "Actual": actuals,
        "GARCH_Pred": forecasts,
    }).set_index("Date")


def backtest_xgboost(
    df: pd.DataFrame,
    ticker: str,
    best_params: dict | None = None,
    initial_train_size: int = 756,
    step_size: int = 21,
) -> pd.DataFrame:
    if best_params is None:
        best_params = {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

    X, y, _ = prepare_xy(df)
    n = len(X)

    forecasts = []
    actuals = []
    dates = []

    for start in range(initial_train_size, n, step_size):
        end = min(start + step_size, n)
        X_tr = X.iloc[:start]
        y_tr = y.iloc[:start]
        X_te = X.iloc[start:end]
        y_te = y.iloc[start:end]

        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            verbosity=0,
            **best_params,
        )
        model.fit(X_tr, y_tr, verbose=False)
        preds = model.predict(X_te)

        forecasts.extend(preds.tolist())
        actuals.extend(y_te.tolist())
        dates.extend(X_te.index.tolist())

    return pd.DataFrame({
        "Date": dates,
        "Actual": actuals,
        "XGB_Pred": forecasts,
    }).set_index("Date")


def run_full_backtest(
    df: pd.DataFrame,
    ticker: str,
    har_params: dict,
    garch_params: dict,
    xgb_params: dict,
    initial_train_size: int = 756,
    step_size: int = 21,
) -> pd.DataFrame:
    print(f"Running backtest for {ticker}...")

    har_df = backtest_har(
        df,
        ticker,
        initial_train_size=initial_train_size,
        step_size=step_size,
        **har_params,
    )
    garch_df = backtest_garch(
        df,
        ticker,
        initial_train_size=initial_train_size,
        step_size=step_size,
        **garch_params,
    )
    xgb_df = backtest_xgboost(
        df,
        ticker,
        best_params=xgb_params,
        initial_train_size=initial_train_size,
        step_size=step_size,
    )

    merged = har_df.join(garch_df[["GARCH_Pred"]], how="left")
    merged = merged.join(xgb_df[["XGB_Pred"]], how="left")
    return merged


def compute_all_metrics(backtest_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = backtest_df.dropna(subset=["Actual"]).copy()
    rows = []

    for col, label in [("HAR_Pred", "HAR"), ("GARCH_Pred", "GARCH"), ("XGB_Pred", "XGBoost")]:
        if col in df.columns and df[col].notna().sum() > 0:
            valid = df[["Actual", col]].dropna()
            m = _compute_metrics(valid["Actual"].values, valid[col].values, label=label)
            m["Ticker"] = ticker
            rows.append(m)

    return pd.DataFrame(rows)
