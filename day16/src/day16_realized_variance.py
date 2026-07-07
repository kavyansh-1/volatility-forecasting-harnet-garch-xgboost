# ─────────────────────────────────────────────────────────────
# day16_realized_variance.py
# Compute realized variance (RV) and bipower variation (BV)
# from intraday 5-minute log returns.
#
# REALIZED VARIANCE — THEORETICAL FOUNDATION
# ─────────────────────────────────────────────────────────────
# Under a continuous-time diffusion process for log-prices:
#   d log(P_t) = mu_t dt + sigma_t dW_t + J_t dN_t
#
# where:
#   mu_t    = drift
#   sigma_t = diffusion (continuous) volatility
#   J_t dN_t = jump component (rare large moves)
#
# The QUADRATIC VARIATION over [0,T] is:
#   [log P]_T = int_0^T sigma_t^2 dt + sum of J_t^2
#
# Realized Variance ESTIMATES this quadratic variation:
#   RV_t = sum_{i=1}^{M} r_{t,i}^2
#   → [log P]_T  as M → ∞ (sampling frequency → ∞)
#
# RV captures BOTH the continuous diffusion component AND jumps.
#
# BIPOWER VARIATION — JUMP-ROBUST ESTIMATOR
# ─────────────────────────────────────────────────────────────
# Barndorff-Nielsen & Shephard (2004) showed that products of
# ADJACENT absolute returns converge to ONLY the continuous part:
#
#   BV_t = (pi/2) * sum_{i=2}^{M} |r_{t,i}| * |r_{t,i-1}|
#   → int_0^T sigma_t^2 dt  (no jumps, even if present)
#
# WHY DOES BV EXCLUDE JUMPS?
# A jump at time i affects r_{t,i} but not r_{t,i-1}.
# So the product |r_i| * |r_{i-1}| cannot be large for BOTH
# if there is a single jump. Only the diffusion component
# produces persistent large products of adjacent returns.
#
# KEY RELATIONSHIP:
#   RV = BV + Jump Variation
#   Jump Variation = RV - BV  (positive when jumps occurred)
# ─────────────────────────────────────────────────────────────

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS      = ["SPY", "QQQ", "AAPL"]
ANNUALISE    = 252          # trading days per year
MU_RATIO     = np.pi / 2   # BV scaling constant = E[|Z|]^{-2} for Z~N(0,1)


def compute_daily_rv(intra_df: pd.DataFrame) -> pd.Series:
    """
    Realized Variance: sum of squared 5-minute log-returns per day.

    RV_t = sum_{i=1}^{M} r_{t,i}^2

    Units: daily variance (not annualised here — annualise downstream).
    """
    rv_daily = (
        intra_df["log_ret"] ** 2
    ).groupby(intra_df["date"]).sum()

    rv_daily.index = pd.to_datetime(rv_daily.index)
    rv_daily.name  = "rv_5min"
    return rv_daily


def compute_daily_bv(intra_df: pd.DataFrame) -> pd.Series:
    """
    Bipower Variation: scaled sum of products of adjacent |returns| per day.

    BV_t = (pi/2) * sum_{i=2}^{M} |r_{t,i}| * |r_{t,i-1}|

    The scaling factor pi/2 comes from:
        E[|r_i| * |r_{i-1}|] = (pi/2) * sigma^2 * dt
    under the null of no jumps (Gaussian increments).
    Dividing the sum by (pi/2) cancels this scaling to give an
    unbiased estimate of integrated variance.

    The product of adjacent returns ensures a single jump at time i
    does NOT make BV explode — it would require large |r_i| AND
    large |r_{i-1}|, which jump processes rarely produce.
    """
    abs_ret = intra_df["log_ret"].abs()

    def bv_for_day(group):
        a = group.values
        if len(a) < 2:
            return np.nan
        # Product of adjacent absolute returns, summed
        return MU_RATIO * np.sum(a[1:] * a[:-1])

    bv_daily = abs_ret.groupby(intra_df["date"]).apply(bv_for_day)
    bv_daily.index = pd.to_datetime(bv_daily.index)
    bv_daily.name  = "bv_5min"
    return bv_daily


def compute_rk(intra_df: pd.DataFrame,
               bandwidth: int = 8) -> pd.Series:
    """
    Realized Kernel (Barndorff-Nielsen et al., 2008).
    A noise-robust estimator that uses auto-covariances
    at multiple lags, weighted by a Parzen kernel.

    RK = sum_{h=-H}^{H} k(h/H) * gamma_h
    gamma_h = sum_{t=|h|+1}^{M} r_t * r_{t-|h|}

    The Parzen kernel k(x):
        k(x) = 1 - 6x^2 + 6x^3  for |x| ≤ 0.5
        k(x) = 2*(1-x)^3         for 0.5 < |x| ≤ 1

    When microstructure noise is present, RV is biased upward
    (noise inflates the sum of squared returns). RK corrects this
    by including negative-lag autocovariance terms that absorb
    the noise contribution.

    Note: On our synthetic data (no microstructure noise),
    RK ≈ RV. On real tick data, RK < RV.
    """
    def parzen_kernel(x: np.ndarray) -> np.ndarray:
        k = np.zeros_like(x, dtype=float)
        mask1 = np.abs(x) <= 0.5
        mask2 = (np.abs(x) > 0.5) & (np.abs(x) <= 1.0)
        k[mask1] = 1 - 6*x[mask1]**2 + 6*np.abs(x[mask1])**3
        k[mask2] = 2 * (1 - np.abs(x[mask2]))**3
        return k

    def rk_for_day(group):
        r = group.values
        n = len(r)
        if n < bandwidth + 2:
            return np.sum(r**2)   # fallback to RV if too few bars

        lags = np.arange(-bandwidth, bandwidth + 1)
        w    = parzen_kernel(lags / (bandwidth + 1))

        rk = 0.0
        for h, wh in zip(lags, w):
            if h == 0:
                rk += wh * np.sum(r**2)
            elif h > 0:
                rk += wh * np.sum(r[h:] * r[:-h])
            else:
                rk += wh * np.sum(r[:h] * r[-h:])
        return max(rk, 0.0)

    rk_daily = intra_df["log_ret"].groupby(intra_df["date"]).apply(rk_for_day)
    rk_daily.index = pd.to_datetime(rk_daily.index)
    rk_daily.name  = "rk_5min"
    return rk_daily


def compute_all_estimators(intra_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RV, BV, RK and their derived quantities in one call.

    Derived quantities:
        jump_var   = max(RV - BV, 0)    — estimated jump variance
        jump_ratio = jump_var / RV      — fraction of total var from jumps
        cont_var   = BV                 — continuous diffusion variance
        rv_ann     = RV * 252           — annualised daily RV
        bv_ann     = BV * 252           — annualised daily BV
        rk_ann     = RK * 252           — annualised daily RK
    """
    rv = compute_daily_rv(intra_df)
    bv = compute_daily_bv(intra_df)
    rk = compute_rk(intra_df)

    df = pd.DataFrame({"rv_5min": rv, "bv_5min": bv, "rk_5min": rk})

    # Jump component: positive part of (RV - BV)
    df["jump_var"]   = np.maximum(df["rv_5min"] - df["bv_5min"], 0.0)
    df["jump_ratio"] = df["jump_var"] / (df["rv_5min"] + 1e-12)
    df["cont_var"]   = df["bv_5min"]

    # Annualised versions (consistent with Days 1-15)
    for col in ["rv_5min", "bv_5min", "rk_5min"]:
        df[f"{col}_ann"] = df[col] * ANNUALISE

    return df


def run_rv_estimation(intraday_data: dict) -> dict:
    """Compute all RV estimators for all tickers."""
    print(f"\n{'='*55}")
    print("  DAY 16 — Realized Variance Estimation")
    print(f"{'='*55}")

    results = {}
    for ticker, intra in intraday_data.items():
        print(f"\n  {ticker}:")
        df = compute_all_estimators(intra)
        results[ticker] = df

        # Summary statistics
        print(f"    Mean RV  (ann) : {df['rv_5min_ann'].mean():.4f}")
        print(f"    Mean BV  (ann) : {df['bv_5min_ann'].mean():.4f}")
        print(f"    Mean RK  (ann) : {df['rk_5min_ann'].mean():.4f}")
        print(f"    Mean jump ratio: {df['jump_ratio'].mean():.4f}")
        print(f"    Max  jump ratio: {df['jump_ratio'].max():.4f}")

        out = os.path.join(OUT_DIR, f"day16_rv_estimates_{ticker}.csv")
        df.to_csv(out)
        print(f"    Saved → {out}")

    return results