# ─────────────────────────────────────────────────────────────
# day09_stability.py
# Model parameter stability analysis.
#
# WHAT IS STABILITY ANALYSIS?
# ─────────────────────────────────────────────────────────────
# A model's parameters (Ridge coefficients, XGB feature
# importances) should ideally be STABLE over time.
# If they drift wildly from one refitting to the next:
#   - The model has found spurious patterns in recent data
#   - It will behave unpredictably when markets change
#   - Regularisation may be too weak
#
# We track how model parameters change across walk-forward
# windows and compute:
#   - Coefficient drift: L2 norm of (coef_t - coef_{t-1})
#   - Feature rank stability: do the top-5 features stay consistent?
#   - Prediction correlation: are adjacent window forecasts correlated?
# ─────────────────────────────────────────────────────────────

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model  import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from xgboost               import XGBRegressor
from scipy.stats           import spearmanr

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUT_DIR  = os.path.join(BASE_DIR, "output")

TICKERS       = ["SPY", "QQQ", "AAPL"]
INITIAL_TRAIN = 756
STEP_SIZE     = 21
ALPHA_HAR     = 10.0


def build_features_and_target(df: pd.DataFrame) -> tuple:
    """Same feature set as day09_walk_forward.py for consistency."""
    rv_1d = df["log_return"] ** 2 * 252
    feats = pd.DataFrame(index=df.index)

    for k in [1, 2, 3, 5, 10, 21]:
        feats[f"rv_lag_{k}"] = rv_1d.shift(k)
    for k in [1, 2, 3, 5]:
        feats[f"ret_lag_{k}"] = df["log_return"].shift(k)
    for col in ["rv_rolling_5d", "rv_rolling_21d"]:
        if col in df.columns:
            feats[col] = df[col].shift(1)
    feats["rv_ewm_10"]    = rv_1d.shift(1).ewm(span=10, min_periods=5).mean()
    feats["abs_ret_lag1"] = df["log_return"].abs().shift(1)

    target   = rv_1d.shift(-1)
    combined = pd.concat([feats, target.rename("target")],
                          axis=1).dropna()
    return (combined.drop(columns=["target"]),
            combined["target"])


def track_har_coefficients(X: pd.DataFrame,
                            y: pd.Series) -> pd.DataFrame:
    """
    Re-fit HAR-Ridge at each walk-forward window.
    Record the coefficient vector at each step.

    Returns DataFrame where each row = one window,
    each column = one coefficient value.
    This lets us plot how each coefficient evolves over time.
    """
    har_cols = [c for c in ["rv_lag_1", "rv_lag_5", "rv_lag_21"]
                if c in X.columns]
    X_har = X[har_cols]
    n     = len(X_har)
    rows  = []

    for start in range(INITIAL_TRAIN, n - STEP_SIZE, STEP_SIZE):
        X_tr = X_har.iloc[:start]
        y_tr = y.iloc[:start]

        pipe = Pipeline([
            ("sc",    StandardScaler()),
            ("ridge", Ridge(alpha=ALPHA_HAR)),
        ])
        pipe.fit(X_tr, y_tr)

        coefs     = pipe.named_steps["ridge"].coef_
        intercept = pipe.named_steps["ridge"].intercept_

        row = {
            "window_end" : X_har.index[start - 1],
            "n_train"    : start,
            "intercept"  : intercept,
        }
        for name, coef in zip(har_cols, coefs):
            row[f"coef_{name}"] = coef

        rows.append(row)

    return pd.DataFrame(rows)


def track_xgb_importance(X: pd.DataFrame,
                          y: pd.Series,
                          params: dict = None) -> pd.DataFrame:
    """
    Re-fit XGBoost at each walk-forward window.
    Record top feature importances (gain) at each step.

    Feature importance stability is measured by Spearman rank
    correlation between adjacent windows — high correlation means
    the model consistently uses the same features.
    """
    if params is None:
        params = {
            "n_estimators" : 100,
            "max_depth"    : 3,
            "learning_rate": 0.05,
            "subsample"    : 0.8,
        }

    n    = len(X)
    rows = []

    for start in range(INITIAL_TRAIN, n - STEP_SIZE, STEP_SIZE):
        X_tr = X.iloc[:start]
        y_tr = y.iloc[:start]

        model = XGBRegressor(
            objective    = "reg:squarederror",
            tree_method  = "hist",
            random_state = 42,
            verbosity    = 0,
            **params,
        )
        model.fit(X_tr, y_tr)

        imp = pd.Series(
            model.feature_importances_,
            index=X.columns
        )
        row = {"window_end": X.index[start - 1], "n_train": start}
        row.update(imp.to_dict())
        rows.append(row)

    return pd.DataFrame(rows)


def compute_drift_metrics(coef_df: pd.DataFrame,
                           coef_cols: list) -> pd.DataFrame:
    """
    Compute parameter drift metrics across consecutive windows.

    Metrics per window transition:
        l2_drift     : L2 norm of (coef_t - coef_{t-1})
                       Large = big jump in parameters
        cosine_sim   : cosine similarity of adjacent coef vectors
                       1.0 = pointing same direction (very stable)
                       0.0 = orthogonal (completely different)
        max_abs_change: max absolute change in any single coefficient
    """
    rows = []
    coef_matrix = coef_df[coef_cols].values

    for i in range(1, len(coef_matrix)):
        delta      = coef_matrix[i] - coef_matrix[i - 1]
        l2_drift   = np.linalg.norm(delta)

        # Cosine similarity
        v1 = coef_matrix[i - 1]
        v2 = coef_matrix[i]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_sim = float(np.dot(v1, v2) / denom) if denom > 1e-10 else 0.0

        rows.append({
            "window"        : i,
            "window_end"    : coef_df["window_end"].iloc[i],
            "l2_drift"      : round(l2_drift,       6),
            "cosine_sim"    : round(cos_sim,         6),
            "max_abs_change": round(np.abs(delta).max(), 6),
        })

    return pd.DataFrame(rows)


def compute_importance_stability(imp_df: pd.DataFrame,
                                  feature_cols: list,
                                  top_k: int = 5) -> pd.DataFrame:
    """
    Measure XGBoost feature importance stability using Spearman
    rank correlation between adjacent windows.

    Spearman correlation of 1.0 = same feature ranking in both windows.
    Spearman correlation of 0.0 = completely different ranking.
    """
    rows = []
    imp_matrix = imp_df[feature_cols].values

    for i in range(1, len(imp_matrix)):
        rho, p = spearmanr(imp_matrix[i - 1], imp_matrix[i])
        rows.append({
            "window"      : i,
            "window_end"  : imp_df["window_end"].iloc[i],
            "spearman_rho": round(float(rho), 4),
            "p_value"     : round(float(p),   4),
        })

    return pd.DataFrame(rows)


def run_stability_analysis() -> dict:
    """
    Run full stability analysis for all tickers.
    Saves coefficient history, importance history,
    and drift metrics.
    """
    print(f"\n{'='*55}")
    print("  DAY 9 — Model Stability Analysis")
    print(f"{'='*55}")

    results = {}

    for ticker in TICKERS:
        path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
        if not os.path.exists(path):
            print(f"\n  ⚠ {path} not found")
            continue

        print(f"\n  {ticker}:")
        df   = pd.read_csv(path, index_col="Date", parse_dates=True)
        X, y = build_features_and_target(df)

        # HAR coefficient tracking
        print("    Tracking HAR coefficients...")
        coef_df   = track_har_coefficients(X, y)
        coef_cols = [c for c in coef_df.columns
                     if c.startswith("coef_")]
        drift_df  = compute_drift_metrics(coef_df, coef_cols)

        # XGBoost importance tracking
        print("    Tracking XGB importances...")
        imp_df      = track_xgb_importance(X, y)
        feat_cols   = [c for c in imp_df.columns
                       if c not in ["window_end", "n_train"]]
        stab_df     = compute_importance_stability(imp_df, feat_cols)

        # Print stability summary
        print(f"    HAR coef drift   : "
              f"mean={drift_df['l2_drift'].mean():.4f}  "
              f"max={drift_df['l2_drift'].max():.4f}")
        print(f"    HAR cosine sim   : "
              f"mean={drift_df['cosine_sim'].mean():.4f}")
        print(f"    XGB Spearman rho : "
              f"mean={stab_df['spearman_rho'].mean():.4f}")

        # Save per-ticker files
        coef_df.to_csv(
            os.path.join(OUT_DIR, f"day09_har_coefs_{ticker}.csv"),
            index=False
        )
        drift_df.to_csv(
            os.path.join(OUT_DIR, f"day09_har_drift_{ticker}.csv"),
            index=False
        )
        imp_df.to_csv(
            os.path.join(OUT_DIR, f"day09_xgb_importance_{ticker}.csv"),
            index=False
        )
        stab_df.to_csv(
            os.path.join(OUT_DIR, f"day09_xgb_stability_{ticker}.csv"),
            index=False
        )

        results[ticker] = {
            "coef_df"   : coef_df,
            "drift_df"  : drift_df,
            "imp_df"    : imp_df,
            "stab_df"   : stab_df,
            "coef_cols" : coef_cols,
            "feat_cols" : feat_cols,
        }

    print(f"\n  ✓ Stability files saved to output/")
    return results