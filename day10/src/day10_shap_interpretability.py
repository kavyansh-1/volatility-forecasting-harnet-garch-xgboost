# ─────────────────────────────────────────────────────────────
# day10_shap_interpretability.py
# SHAP (SHapley Additive exPlanations) analysis for the
# XGBoost volatility model.
#
# WHY SHAP INSTEAD OF JUST feature_importances_?
# ─────────────────────────────────────────────────────────────
# XGBoost's built-in feature_importances_ (used in Days 4, 7, 9)
# tells you AVERAGE importance across the whole dataset, but:
#   - It doesn't tell you the DIRECTION of effect (does high
#     rv_lag_1 push the forecast UP or DOWN?)
#   - It doesn't tell you if the effect is consistent or
#     depends on the values of OTHER features (interactions)
#   - It can't explain a SINGLE prediction (why did the model
#     forecast 18% vol for THIS specific day?)
#
# SHAP solves all three. Based on cooperative game theory
# (Shapley values), it fairly attributes each prediction's
# deviation from the average to each input feature.
#
# KEY SHAP OUTPUTS USED HERE:
#   1. Summary plot   : feature importance + direction in one chart
#   2. Dependence plot: how does the SHAP value change as a
#                        specific feature's value changes?
#   3. Waterfall plot : explains ONE individual prediction
#      (e.g. "why did the model predict high vol on this exact day?")
# ─────────────────────────────────────────────────────────────

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("  ⚠ shap not installed. Run: pip install shap")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUT_DIR  = os.path.join(BASE_DIR, "output")

TICKERS   = ["SPY", "QQQ", "AAPL"]
TEST_SIZE = 500

XGB_PARAMS = {
    "n_estimators"    : 200,
    "max_depth"       : 3,
    "learning_rate"   : 0.05,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda"      : 2.0,
}


def build_features_and_target(df: pd.DataFrame) -> tuple:
    """Same feature set used throughout Days 4-9 for consistency."""
    rv_1d = df["log_return"] ** 2 * 252
    feats = pd.DataFrame(index=df.index)

    for k in [1, 2, 3, 5, 10, 21]:
        feats[f"rv_lag_{k}"] = rv_1d.shift(k)
    for k in [1, 2, 3, 5]:
        feats[f"ret_lag_{k}"] = df["log_return"].shift(k)
    for col in ["rv_rolling_5d", "rv_rolling_21d", "rv_rolling_63d"]:
        if col in df.columns:
            feats[col] = df[col].shift(1)
    for col in ["park_vol_5d", "gk_vol_5d"]:
        if col in df.columns:
            feats[col] = df[col].shift(1)

    feats["rv_ewm_10"]    = rv_1d.shift(1).ewm(span=10, min_periods=5).mean()
    feats["abs_ret_lag1"] = df["log_return"].abs().shift(1)

    target   = rv_1d.shift(-1)
    combined = pd.concat([feats, target.rename("target")],
                          axis=1).dropna()
    return combined.drop(columns=["target"]), combined["target"]


def fit_xgb_for_shap(X: pd.DataFrame,
                      y: pd.Series) -> tuple:
    """
    Fit a single XGBoost model on the training period.
    Returns (model, X_train, X_test, y_test).
    """
    n    = len(X)
    split = n - TEST_SIZE

    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(
        objective    = "reg:squarederror",
        tree_method  = "hist",
        random_state = 42,
        verbosity    = 0,
        **XGB_PARAMS,
    )
    model.fit(X_tr, y_tr)

    return model, X_tr, X_te, y_te


def compute_shap_values(model, X_test: pd.DataFrame):
    """
    Compute SHAP values using TreeExplainer — the fast, exact
    algorithm for tree-based models (no approximation needed,
    unlike KernelExplainer used for black-box models).

    Returns the shap.Explanation object containing:
        .values     : (n_samples, n_features) SHAP values
        .base_values: expected model output (average prediction)
        .data       : the input feature values
    """
    if not SHAP_AVAILABLE:
        return None

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    return shap_values


def summarise_shap_importance(shap_values,
                               feature_names: list) -> pd.DataFrame:
    """
    Convert raw SHAP values into a ranked importance table.

    mean_abs_shap = mean(|SHAP value|) across all test samples.
    This is SHAP's equivalent of feature_importances_, but
    derived directly from the additive explanation rather than
    from internal tree split statistics — generally considered
    more reliable, especially with correlated features.
    """
    abs_vals = np.abs(shap_values.values)
    mean_abs = abs_vals.mean(axis=0)

    # Also compute the average SIGNED shap value: tells us the
    # typical DIRECTION of each feature's effect
    mean_signed = shap_values.values.mean(axis=0)

    df = pd.DataFrame({
        "feature"        : feature_names,
        "mean_abs_shap"  : mean_abs,
        "mean_signed_shap": mean_signed,
    }).sort_values("mean_abs_shap", ascending=False)

    df["direction"] = np.where(
        df["mean_signed_shap"] > 0, "increases forecast",
        "decreases forecast"
    )
    return df.reset_index(drop=True)


def run_shap_for_ticker(ticker: str) -> dict:
    """
    Full SHAP pipeline for one ticker.
    Fits XGBoost, computes SHAP values, builds importance table.
    """
    path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
    if not os.path.exists(path):
        print(f"  ⚠ {path} not found")
        return None

    df   = pd.read_csv(path, index_col="Date", parse_dates=True)
    X, y = build_features_and_target(df)

    model, X_tr, X_te, y_te = fit_xgb_for_shap(X, y)

    if not SHAP_AVAILABLE:
        return {"ticker": ticker, "shap_available": False}

    print(f"  {ticker}: computing SHAP values for "
          f"{len(X_te)} test observations × {X_te.shape[1]} features...")

    shap_values = compute_shap_values(model, X_te)
    importance_df = summarise_shap_importance(
        shap_values, X_te.columns.tolist()
    )

    print(f"    Top 3 features by mean |SHAP|:")
    for _, row in importance_df.head(3).iterrows():
        print(f"      {row['feature']:18s}  "
              f"{row['mean_abs_shap']:.6f}  ({row['direction']})")

    return {
        "ticker"        : ticker,
        "shap_available": True,
        "shap_values"   : shap_values,
        "importance_df" : importance_df,
        "X_test"        : X_te,
        "y_test"        : y_te,
        "model"         : model,
    }


def run_all_shap_analysis() -> dict:
    """Run SHAP analysis for all tickers, save importance CSVs."""
    print(f"\n{'='*55}")
    print("  DAY 10 — SHAP Interpretability")
    print(f"{'='*55}")

    if not SHAP_AVAILABLE:
        print("  ⚠ shap library not available — skipping module")
        return {}

    results = {}
    all_importance = []

    for ticker in TICKERS:
        res = run_shap_for_ticker(ticker)
        if res is None or not res.get("shap_available"):
            continue
        results[ticker] = res

        imp = res["importance_df"].copy()
        imp.insert(0, "Ticker", ticker)
        all_importance.append(imp)

    if all_importance:
        combined = pd.concat(all_importance, ignore_index=True)
        out = os.path.join(OUT_DIR, "day10_shap_importance.csv")
        combined.to_csv(out, index=False)
        print(f"\n  ✓ {out}")

    return results