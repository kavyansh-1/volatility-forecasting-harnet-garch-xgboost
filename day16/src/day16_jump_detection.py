# ─────────────────────────────────────────────────────────────
# day16_jump_detection.py
# Formal statistical tests for price jumps.
#
# WHY DETECT JUMPS?
# ─────────────────────────────────────────────────────────────
# From a forecasting perspective, jumps matter because:
#   1. Jump days have much higher daily RV than diffusion alone
#      would predict — the jump adds a "surprise" component
#   2. Jump variance is NOT persistent — it does NOT autocorrelate
#      the way diffusion variance does. A big jump today does NOT
#      predict a big jump tomorrow (unlike diffusion clustering)
#   3. If you include jumps in your training target (RV = BV + Jump),
#      you're trying to forecast something partially unpredictable
#      BV-only targets produce cleaner forecasting problems
#   4. Jump days often correspond to known macro/earnings events —
#      identifying them allows event-driven analysis
#
# TWO JUMP TESTS IMPLEMENTED:
# ─────────────────────────────────────────────────────────────
# 1. Barndorff-Nielsen & Shephard (2004) ratio test:
#    J_t = (RV_t - BV_t) / (RV_t * sqrt(variance_factor))
#    Under H0 (no jumps): J_t → N(0,1)
#    Reject H0 when |J_t| > z_{alpha/2} (e.g. 2.576 for 99%)
#
# 2. Lee & Mykland (2008) test:
#    Tests EACH intraday return individually for being a jump.
#    L_t,i = r_{t,i} / local_vol
#    where local_vol is estimated from nearby bars.
#    |L_t,i| > threshold → bar i contains a jump.
#    This gives not just "was there a jump today" but
#    "WHEN during the day did the jump occur."
# ─────────────────────────────────────────────────────────────

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS    = ["SPY", "QQQ", "AAPL"]
BARS_PER_DAY = 78
ALPHA      = 0.01    # significance level for jump detection
# Variance factor for BNS test (depends on M and pi)
# Derived in Barndorff-Nielsen & Shephard (2004), Section 3
_ZETA1     = np.pi ** 2 / 4 + np.pi - 5
_MU4_OVER_MU2_SQ = (np.pi / 2) ** 2   # = (pi/2)^2 for adjacent products
PHI        = _ZETA1 * _MU4_OVER_MU2_SQ  # asymptotic variance factor


def bns_jump_test(rv:    pd.Series,
                  bv:    pd.Series,
                  alpha: float = ALPHA) -> pd.DataFrame:
    """
    Barndorff-Nielsen & Shephard (2004) test for daily jumps.

    Test statistic:
        J_t = sqrt(M) * (RV_t - BV_t) / RV_t
              / sqrt(PHI * max(1, BQ_t/BV_t^2))

    Under H0 (no jumps on day t):
        J_t → N(0, 1)

    We use the simplified version (without realised quarticity BQ):
        J_t = (RV_t - BV_t) / sqrt(PHI/M * RV_t^2)

    Critical value: z_{1 - alpha/2} = 2.576 at alpha=0.01

    Returns DataFrame with one row per day:
        J_stat, p_value, jump_detected (bool)
    """
    M  = BARS_PER_DAY
    eps = 1e-12

    diff   = rv - bv
    se     = np.sqrt(PHI / M) * (rv + eps)
    J_stat = diff / se

    p_value = 2 * (1 - stats.norm.cdf(np.abs(J_stat)))
    jump    = (p_value < alpha).astype(int)

    return pd.DataFrame({
        "J_stat"        : J_stat.round(4),
        "p_value"       : p_value.round(6),
        "jump_detected" : jump,
    }, index=rv.index)


def lee_mykland_test(intra_df: pd.DataFrame,
                     alpha:    float = ALPHA,
                     window:   int   = 20) -> pd.DataFrame:
    """
    Lee & Mykland (2008) intraday jump test.

    For each 5-minute return r_{t,i}, compute a LOCAL volatility
    estimate from the preceding `window` bars:
        sigma_local = BV estimate from bars [i-window, i-1]

    Test statistic:
        L_{t,i} = r_{t,i} / sigma_local

    Critical value under H0 (no jump at bar i):
        |L_{t,i}| > C_{n} where C_{n} = sqrt(2 * log(n))
        and n = total number of bars in the sample

    This test identifies the EXACT BAR within the day where a
    jump occurred, not just whether the day had a jump.

    Returns DataFrame with columns:
        datetime, date, bar, L_stat, jump_bar (bool)
    """
    n     = len(intra_df)
    C_n   = np.sqrt(2 * np.log(n))         # critical value
    beta_n= C_n - np.log(np.log(n)) / (2 * C_n)  # finite-sample correction

    abs_ret = intra_df["log_ret"].abs().values
    results = []

    for i in range(window, n):
        # Local BV from preceding `window` bars (rolling window)
        local_abs = abs_ret[i - window : i]
        # BV estimate of local vol: sqrt(pi/2 * sum |r_i||r_{i-1}|)
        local_bv  = (np.pi / 2) * np.sum(local_abs[1:] * local_abs[:-1])
        sigma_loc = np.sqrt(local_bv / (window - 1) + 1e-12)

        L_stat    = abs_ret[i] / sigma_loc
        jump_bar  = int(L_stat > beta_n)

        results.append({
            "datetime"  : intra_df.index[i],
            "date"      : intra_df["date"].iloc[i],
            "bar"       : int(intra_df["bar"].iloc[i]),
            "L_stat"    : round(float(L_stat), 4),
            "jump_bar"  : jump_bar,
        })

    return pd.DataFrame(results)


def aggregate_lm_to_daily(lm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Lee-Mykland intraday results to daily summary.
    For each day: how many bars were flagged as jumps?
    What was the maximum L-stat (most extreme intraday return)?
    """
    agg = lm_df.groupby("date").agg(
        n_jump_bars = ("jump_bar",  "sum"),
        max_L_stat  = ("L_stat",   "max"),
        n_bars      = ("bar",       "count"),
    ).reset_index()
    agg["date"]            = pd.to_datetime(agg["date"])
    agg["jump_day_lm"]     = (agg["n_jump_bars"] > 0).astype(int)
    agg["jump_intensity"]  = agg["n_jump_bars"] / agg["n_bars"]
    return agg.set_index("date")


def run_jump_detection(intraday_data: dict,
                        rv_results:   dict) -> dict:
    """
    Run both jump tests for all tickers. Combine results.
    """
    print(f"\n{'='*55}")
    print("  DAY 16 — Jump Detection")
    print(f"{'='*55}")
    print(f"  BNS test alpha={ALPHA}  Lee-Mykland window=20")

    all_results = {}

    for ticker in TICKERS:
        if ticker not in intraday_data or ticker not in rv_results:
            continue

        print(f"\n  {ticker}:")
        intra = intraday_data[ticker]
        rv_df = rv_results[ticker]

        # ── BNS test ───────────────────────────────────────────
        bns_df = bns_jump_test(rv_df["rv_5min"], rv_df["bv_5min"])
        n_jump_bns = bns_df["jump_detected"].sum()
        pct_bns    = n_jump_bns / len(bns_df) * 100
        print(f"    BNS: {n_jump_bns}/{len(bns_df)} jump days ({pct_bns:.1f}%)")

        # ── Lee-Mykland test ───────────────────────────────────
        lm_intra = lee_mykland_test(intra)
        lm_daily = aggregate_lm_to_daily(lm_intra)
        n_jump_lm = lm_daily["jump_day_lm"].sum()
        pct_lm    = n_jump_lm / len(lm_daily) * 100
        print(f"    LM:  {n_jump_lm}/{len(lm_daily)} jump days ({pct_lm:.1f}%)")

        # Agreement between tests
        common   = bns_df.index.intersection(lm_daily.index)
        both_yes = ((bns_df.loc[common, "jump_detected"] == 1) &
                    (lm_daily.loc[common, "jump_day_lm"] == 1)).sum()
        print(f"    Agreement: {both_yes} days flagged by BOTH tests")

        # Combined results
        combined = rv_df.copy()
        combined = combined.join(bns_df, how="left")
        combined = combined.join(
            lm_daily[["n_jump_bars","max_L_stat","jump_day_lm",
                       "jump_intensity"]],
            how="left"
        )

        out = os.path.join(OUT_DIR, f"day16_jump_results_{ticker}.csv")
        combined.to_csv(out)
        print(f"Saved → {out}")

        all_results[ticker] = {
            "combined_df" : combined,
            "bns_df"      : bns_df,
            "lm_daily"    : lm_daily,
            "lm_intra"    : lm_intra,
        }

    return all_results