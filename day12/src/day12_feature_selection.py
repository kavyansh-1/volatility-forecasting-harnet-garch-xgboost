# ─────────────────────────────────────────────────────────────
# day12_feature_selection.py
# Tests whether the new Day 12 features actually help.
# Uses four complementary feature selection methods:
#
#   1. Pearson correlation with target (fast, linear only)
#   2. Mutual information (captures non-linear relationships)
#   3. XGBoost feature importance (tree-based, accounts for
#      feature interactions, same model used for forecasting)
#   4. RFECV (Recursive Feature Elimination with CV) on Ridge
#      — sequentially removes the least useful features,
#      uses cross-validation to pick the optimal feature count
#
# IMPORTANT: all selection methods use TRAINING data only.
# The test set is NEVER touched during selection.
# Cross-validation inside RFECV uses TimeSeriesSplit (no leakage).
# ─────────────────────────────────────────────────────────────

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    mutual_info_regression,
    RFECV,
    SelectKBest,
    f_regression,
)
from sklearn.linear_model    import Ridge
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from xgboost                 import XGBRegressor

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR   = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS   = ["SPY", "QQQ", "AAPL"]
TEST_SIZE = 500


def build_target(df: pd.DataFrame) -> pd.Series:
    """Next-day annualised realized variance (same as Days 4-9)."""
    return (df["log_return"] ** 2 * 252).shift(-1)


def prepare_xy(df: pd.DataFrame) -> tuple:
    """
    Build feature matrix X and target y from the fully enriched df.
    Excludes raw OHLCV columns (Open, High, Low, Close, Volume)
    and the original log_return (it becomes lagged features) from X.
    All remaining numeric columns with shift >= 1 are valid features.
    """
    target   = build_target(df)
    EXCLUDE  = {"Open", "High", "Low", "Close", "Volume",
                "log_return", "simple_return"}
    feat_cols = [c for c in df.columns
                 if c not in EXCLUDE
                 and pd.api.types.is_numeric_dtype(df[c])]

    combined = pd.concat([df[feat_cols],
                           target.rename("target")], axis=1).dropna()
    X = combined.drop(columns=["target"])
    y = combined["target"]

    n     = len(X)
    split = n - TEST_SIZE
    return (X.iloc[:split], X.iloc[split:],
            y.iloc[:split], y.iloc[split:])


def pearson_selection(X_train: pd.DataFrame,
                       y_train:  pd.Series,
                       top_k:    int = 20) -> pd.DataFrame:
    """
    Rank features by absolute Pearson correlation with the target.
    Fast but only captures LINEAR associations.
    """
    corrs = X_train.corrwith(y_train).abs().sort_values(ascending=False)
    return pd.DataFrame({
        "feature"         : corrs.index,
        "pearson_abs_corr": corrs.values,
        "rank_pearson"    : range(1, len(corrs) + 1),
    }).head(top_k)


def mutual_info_selection(X_train: pd.DataFrame,
                           y_train:  pd.Series,
                           top_k:    int = 20) -> pd.DataFrame:
    """
    Rank features by mutual information with the target.
    Captures NONLINEAR associations — complementary to Pearson.
    mutual_info_regression uses a k-nearest-neighbours estimator
    (Kraskov et al., 2004) — no distributional assumptions.
    """
    mi = mutual_info_regression(
        X_train.fillna(0), y_train,
        random_state=42
    )
    df = pd.DataFrame({
        "feature"   : X_train.columns,
        "mutual_info": mi,
    }).sort_values("mutual_info", ascending=False)
    df["rank_mi"] = range(1, len(df) + 1)
    return df.head(top_k)


def xgb_importance_selection(X_train: pd.DataFrame,
                               y_train:  pd.Series,
                               top_k:    int = 20) -> pd.DataFrame:
    """
    Rank features by XGBoost gain-based importance.
    Accounts for feature interactions and non-linearity.
    Uses a lightweight XGBoost (fast, not tuned) — the goal
    is ranking, not optimal forecasting performance.
    """
    model = XGBRegressor(
        n_estimators  = 100,
        max_depth     = 3,
        learning_rate = 0.1,
        subsample     = 0.8,
        random_state  = 42,
        verbosity     = 0,
    )
    model.fit(X_train.fillna(0), y_train)
    imp = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    df = pd.DataFrame({
        "feature"    : imp.index,
        "xgb_gain"   : imp.values,
        "rank_xgb"   : range(1, len(imp) + 1),
    })
    return df.head(top_k)


def rfecv_selection(X_train: pd.DataFrame,
                     y_train:  pd.Series) -> dict:
    """
    Recursive Feature Elimination with Cross-Validation on Ridge.
    Starts with all features and removes the least important one
    at each step. TimeSeriesSplit ensures no temporal leakage.

    Returns the optimal feature subset and per-step CV scores.
    This is the most rigorous method but also the slowest — it
    fits (n_features) × (cv_splits) models to find the optimal count.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    pipe = Pipeline([
        ("sc",    StandardScaler()),
        ("ridge", Ridge(alpha=10.0)),
    ])

    rfecv = RFECV(
        estimator  = pipe,
        step       = 1,
        cv         = tscv,
        scoring    = "neg_root_mean_squared_error",
        min_features_to_select = 5,
        n_jobs     = -1,
        importance_getter = lambda est: est.named_steps["ridge"].coef_,
    )

    X_clean = X_train.fillna(0)
    rfecv.fit(X_clean, y_train)

    selected = X_train.columns[rfecv.support_].tolist()
    cv_scores = -rfecv.cv_results_["mean_test_score"]   # flip sign

    return {
        "selected_features" : selected,
        "n_selected"        : rfecv.n_features_,
        "cv_scores"         : cv_scores,
        "n_features_tried"  : len(X_train.columns),
    }


def run_feature_selection(macro_dfs: dict) -> dict:
    """
    Run all four selection methods for all tickers.
    Saves combined importance table and RFECV selected feature list.
    """
    print(f"\n{'='*55}")
    print("  DAY 12 — Feature Selection")
    print(f"{'='*55}")
    print("  Methods: Pearson | MI | XGB Importance | RFECV")

    all_results   = {}
    summary_rows  = []

    for ticker in TICKERS:
        if ticker not in macro_dfs:
            continue

        df = macro_dfs[ticker]
        X_tr, X_te, y_tr, y_te = prepare_xy(df)

        print(f"\n  {ticker}: {X_tr.shape[1]} total features, "
              f"{len(y_tr)} training rows")

        pearson_df = pearson_selection(X_tr, y_tr, top_k=20)
        mi_df      = mutual_info_selection(X_tr, y_tr, top_k=20)
        xgb_df     = xgb_importance_selection(X_tr, y_tr, top_k=20)

        print(f"    Running RFECV (slowest step)...")
        rfecv_res  = rfecv_selection(X_tr, y_tr)
        n_sel = rfecv_res["n_selected"]
        print(f"    RFECV optimal features: {n_sel}/{X_tr.shape[1]}")
        print(f"    Top 5 by RFECV: {rfecv_res['selected_features'][:5]}")

        # Merge all rankings into one master table
        merged = (
            pearson_df.set_index("feature")[["pearson_abs_corr", "rank_pearson"]]
            .join(mi_df.set_index("feature")[["mutual_info", "rank_mi"]], how="outer")
            .join(xgb_df.set_index("feature")[["xgb_gain", "rank_xgb"]], how="outer")
        )
        merged["rfecv_selected"] = merged.index.isin(
            rfecv_res["selected_features"]
        ).astype(int)
        merged["avg_rank"] = merged[
            ["rank_pearson", "rank_mi", "rank_xgb"]
        ].mean(axis=1)
        merged = merged.sort_values("avg_rank").reset_index()
        merged.insert(0, "Ticker", ticker)

        out = os.path.join(OUT_DIR, f"day12_feature_importance_{ticker}.csv")
        merged.to_csv(out, index=False)

        # RFECV selected feature list
        rfecv_df = pd.DataFrame({
            "Ticker" : ticker,
            "feature": rfecv_res["selected_features"],
        })
        rfecv_df.to_csv(
            os.path.join(OUT_DIR, f"day12_rfecv_features_{ticker}.csv"),
            index=False
        )

        summary_rows.append({
            "Ticker"           : ticker,
            "Total_features"   : X_tr.shape[1],
            "RFECV_selected"   : n_sel,
            "Top_Pearson"      : pearson_df.iloc[0]["feature"],
            "Top_MI"           : mi_df.iloc[0]["feature"],
            "Top_XGB"          : xgb_df.iloc[0]["feature"],
        })

        all_results[ticker] = {
            "pearson_df" : pearson_df,
            "mi_df"      : mi_df,
            "xgb_df"     : xgb_df,
            "rfecv_res"  : rfecv_res,
            "merged_df"  : merged,
            "X_train"    : X_tr,
            "X_test"     : X_te,
            "y_train"    : y_tr,
            "y_test"     : y_te,
        }

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(OUT_DIR, "day12_feature_selection_summary.csv"),
        index=False
    )
    print(f"\n  OK Feature importance CSVs saved per ticker")
    print(f"  OK day12_feature_selection_summary.csv")
    print(summary_df.to_string(index=False))

    return all_results


if __name__ == "__main__":
    print("  Run via day12_run_all.py")