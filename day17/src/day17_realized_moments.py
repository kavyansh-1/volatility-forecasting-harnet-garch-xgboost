# ─────────────────────────────────────────────────────────────
# day17_realized_moments.py
# Compute realized skewness and kurtosis from intraday returns.
# Use them as predictors for next-day volatility.
#
# WHY HIGHER-ORDER REALIZED MOMENTS?
# ─────────────────────────────────────────────────────────────
# Realized Variance (Day 16) = second moment of intraday returns.
# Realized Skewness  = third moment — measures intraday asymmetry.
# Realized Kurtosis  = fourth moment — measures intraday tail risk.
#
# ECONOMIC MOTIVATION:
#
#   Realized Skewness (RS):
#     A day where returns were mostly positive but had ONE large
#     negative spike has NEGATIVE RS. This "left-tail event"
#     within the day often PRECEDES elevated vol the next day
#     (market participants react to the intraday crash).
#     RS is a same-day early warning for tomorrow's vol.
#
#   Realized Kurtosis (RK):
#     A day with many small returns but several large spikes
#     has HIGH RK — "fat-tailed intraday distribution."
#     High RK (excess kurtosis) correlates with jump activity
#     (which we measured in Day 16 with BNS/Lee-Mykland).
#     High-kurtosis days tend to have elevated future vol
#     because the tail events signal structural uncertainty.
#
# FORMULAS:
#   RS_t = M^{1/2} * sum(r_{t,i}^3) / RV_t^{3/2}
#   RK_t = M * sum(r_{t,i}^4) / RV_t^2
#
#   Scaling by M makes RS/RK dimensionless and comparable
#   across days with different M (Neuberger, 2012).
# ─────────────────────────────────────────────────────────────

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model  import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.model_selection import TimeSeriesSplit

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS      = ["SPY", "QQQ", "AAPL"]
BARS_PER_DAY = 78
TEST_SIZE    = 500
ANNUALISE    = 252


def compute_realized_moments(intra_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute realized variance, skewness, and kurtosis per day.

    RS_t = sqrt(M) * sum(r_{t,i}^3) / RV_t^{3/2}
    RK_t = M * sum(r_{t,i}^4) / RV_t^2
    RS_signed = sign(sum(r^3)) * |RS|  -- directional

    Note on normalisation:
        sqrt(M) in RS and M in RK ensure the statistics have
        mean 0 and known variance under Gaussian innovations.
        This makes RS and RK comparable across different M.
    """
    def moments_for_day(group):
        r   = group.values
        M   = len(r)
        if M < 5:
            return pd.Series({
                "rv"    : np.nan,
                "rs"    : np.nan,
                "rk"    : np.nan,
                "rs_raw": np.nan,
                "rk_raw": np.nan,
            })
        rv     = np.sum(r**2)
        rs_raw = np.sum(r**3)
        rk_raw = np.sum(r**4)

        # Normalised moments (Neuberger, 2012 scaling)
        rs  = (np.sqrt(M) * rs_raw / (rv**(1.5) + 1e-20)
               if rv > 1e-20 else 0.0)
        rk  = (M * rk_raw / (rv**2 + 1e-20)
               if rv > 1e-20 else 0.0)

        return pd.Series({
            "rv"    : rv,
            "rs"    : rs,
            "rk"    : rk,
            "rs_raw": rs_raw,
            "rk_raw": rk_raw,
        })

    mom_df = intra_df["log_ret"].groupby(intra_df["date"]).apply(moments_for_day)
    mom_df.index = pd.to_datetime(mom_df.index)

    # Annualise RV for consistency with Days 4-16
    mom_df["rv_ann"] = mom_df["rv"] * ANNUALISE

    return mom_df


def build_moment_features(daily_df:   pd.DataFrame,
                            moments_df: pd.DataFrame,
                            ticker:     str) -> tuple:
    """
    Build a feature matrix combining:
    1. HAR features (rv lags 1, 5, 21)        — from Day 2 daily data
    2. Realized skewness (rs lag 1)            — from intraday moments
    3. Realized kurtosis (rk lag 1)            — from intraday moments
    4. Rolling rs/rk (5-day and 21-day averages)
    5. Signed RS indicator (was intraday left-skewed?)

    Target: next-day annualised RV (from intraday moments for consistency)
    """
    rv_1d = daily_df["log_return"] ** 2 * ANNUALISE

    # Standard HAR features from daily data
    X = pd.DataFrame(index=daily_df.index)
    X["rv_lag_1"]  = rv_1d.shift(1)
    X["rv_lag_5"]  = rv_1d.shift(1).rolling(5,  min_periods=5).mean()
    X["rv_lag_21"] = rv_1d.shift(1).rolling(21, min_periods=21).mean()

    # Merge in realised moments (intraday data)
    X = X.join(moments_df[["rv_ann", "rs", "rk"]], how="left")

    # Lagged moment features
    X["rs_lag1"]   = X["rs"].shift(1)
    X["rk_lag1"]   = X["rk"].shift(1)
    X["rs_lag5"]   = X["rs"].shift(1).rolling(5,  min_periods=3).mean()
    X["rk_lag5"]   = X["rk"].shift(1).rolling(5,  min_periods=3).mean()
    X["rs_lag21"]  = X["rs"].shift(1).rolling(21, min_periods=10).mean()
    X["rk_lag21"]  = X["rk"].shift(1).rolling(21, min_periods=10).mean()

    # Signed skewness indicator: was yesterday intraday left-skewed?
    # Value: 1 = left-skewed (bearish intraday), 0 = right-skewed
    X["left_skew_flag"] = (X["rs"].shift(1) < 0).astype(float)

    # Interaction: left-skew × rv_lag_1 (amplified effect when both high)
    X["lev_interaction"] = X["left_skew_flag"] * X["rv_lag_1"]

    # Drop temporary columns not meant as features
    X = X.drop(columns=["rv_ann", "rs", "rk"], errors="ignore")

    # Target: next-day intraday RV
    target = moments_df["rv_ann"].shift(-1).reindex(X.index)

    # Fallback target: daily proxy if intraday not available
    target = target.fillna(rv_1d.shift(-1))

    combined = pd.concat([X, target.rename("target")], axis=1).dropna()
    return combined.drop(columns=["target"]), combined["target"]


def moment_augmented_har(X_tr:  pd.DataFrame,
                          y_tr:  pd.Series,
                          X_te:  pd.DataFrame,
                          alpha: float = 10.0) -> np.ndarray:
    """
    HAR-Ridge augmented with realized skewness and kurtosis features.
    Uses the same Ridge pipeline as Days 4-16 for consistency.
    """
    pipe = Pipeline([
        ("sc",    StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])
    pipe.fit(X_tr.fillna(0), y_tr)
    return pipe.predict(X_te.fillna(0))


def run_realized_moments(intraday_data: dict) -> dict:
    """Compute realized moments and test their predictive power."""
    print(f"\n{'='*55}")
    print("  DAY 17 — Realized Moments as Predictors")
    print(f"{'='*55}")

    results  = {}
    met_rows = []

    for ticker in TICKERS:
        if ticker not in intraday_data:
            continue

        path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
        if not os.path.exists(path):
            continue

        print(f"\n  {ticker}:")
        daily_df   = pd.read_csv(path, index_col="Date", parse_dates=True)
        intra_df   = intraday_data[ticker]
        moments_df = compute_realized_moments(intra_df)

        print(f"    Mean RS : {moments_df['rs'].mean():.4f}"
              f"  (negative = avg left-skewed intraday)")
        print(f"    Mean RK : {moments_df['rk'].mean():.4f}"
              f"  (excess kurtosis from intraday fat tails)")

        X, y = build_moment_features(daily_df, moments_df, ticker)
        n     = len(X)
        split = n - TEST_SIZE

        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y.iloc[:split], y.iloc[split:]

        # Model A: HAR only (no moments)
        har_cols = ["rv_lag_1", "rv_lag_5", "rv_lag_21"]
        har_cols = [c for c in har_cols if c in X_tr.columns]
        pipe_base = Pipeline([("sc", StandardScaler()),
                              ("ridge", Ridge(alpha=10.0))])
        pipe_base.fit(X_tr[har_cols].fillna(0), y_tr)
        pred_base = pipe_base.predict(X_te[har_cols].fillna(0))

        # Model B: HAR + realized moments
        pred_aug = moment_augmented_har(X_tr, y_tr, X_te)

        # Metrics
        def rmse(a, b): return np.sqrt(np.mean((a-b)**2))
        def qlike(a, b):
            h = np.maximum(b, 1e-8)
            v = np.maximum(a, 1e-8)
            return np.mean(np.log(h) + v/h)

        y_v = y_te.values
        met_rows.extend([
            {"Ticker": ticker, "Model": "HAR_baseline",
             "RMSE": round(rmse(y_v, pred_base), 8),
             "QLIKE": round(qlike(y_v, pred_base), 6)},
            {"Ticker": ticker, "Model": "HAR+Moments",
             "RMSE": round(rmse(y_v, pred_aug), 8),
             "QLIKE": round(qlike(y_v, pred_aug), 6)},
        ])

        # Save moments
        mom_out = moments_df.copy()
        mom_out.insert(0, "ticker", ticker)
        mom_out.to_csv(
            os.path.join(OUT_DIR, f"day17_realized_moments_{ticker}.csv")
        )

        results[ticker] = {
            "moments_df" : moments_df,
            "X"          : X,
            "y"          : y,
            "pred_base"  : pred_base,
            "pred_aug"   : pred_aug,
            "y_test"     : y_te,
        }

    met_df = pd.DataFrame(met_rows)
    met_df.to_csv(
        os.path.join(OUT_DIR, "day17_moment_metrics.csv"), index=False
    )
    print(f"\n  RMSE comparison (HAR vs HAR+Moments):")
    print(met_df.pivot_table(index="Model", columns="Ticker",
                              values="RMSE").round(8).to_string())
    print(f"\n  [OK] day17_realized_moments_{{ticker}}.csv")
    print(f"  [OK] day17_moment_metrics.csv")
    return results