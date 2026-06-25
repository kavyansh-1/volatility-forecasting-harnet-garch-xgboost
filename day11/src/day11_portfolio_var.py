# ─────────────────────────────────────────────────────────────
# day11_portfolio_var.py
# Combines individual asset volatility forecasts with a
# correlation forecast (static CCC vs dynamic DCC) to produce
# PORTFOLIO-level volatility, VaR, and CVaR — then backtests
# both approaches with Kupiec and Christoffersen tests.
#
# PORTFOLIO VARIANCE FORMULA
# ─────────────────────────────────────────────────────────────
#   sigma_p,t^2 = w' H_t w
#   H_t = D_t R_t D_t     (elementwise: H_ij = sigma_i,t*sigma_j,t*R_ij,t)
#   D_t = diag(sigma_1,t, sigma_2,t, sigma_3,t)
#   w   = portfolio weight vector
#
# CCC (Constant Conditional Correlation):
#   R_t = R_bar  for all t  (full-sample average correlation, fixed)
#   Only the diagonal (individual vols) varies over time.
#
# DCC (Dynamic Conditional Correlation):
#   R_t varies every day per Module 2's recursion.
#   This captures the well-documented stylised fact that
#   correlations tend to RISE during market stress — exactly
#   when diversification benefits matter most.
#
# WHY COMPARE THEM?
# A CCC model that ignores correlation breakdowns during crises
# will systematically UNDERESTIMATE portfolio VaR exactly when
# accurate risk estimates are most critical. This comparison is
# the practical payoff of building the DCC model in Module 2.
# ─────────────────────────────────────────────────────────────

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from day11_realized_covariance import TICKERS, run_realized_covariance
from day11_dcc_correlation     import run_dcc_analysis

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

WEIGHTS   = {"SPY": 1/3, "QQQ": 1/3, "AAPL": 1/3}
TEST_SIZE = 500
ALPHAS    = [0.05, 0.01]


def portfolio_returns(returns: pd.DataFrame,
                      weights: dict) -> pd.Series:
    """Simple linear portfolio return: r_p,t = sum_i(w_i * r_i,t)."""
    w = np.array([weights[t] for t in returns.columns])
    return pd.Series(returns.values @ w, index=returns.index,
                     name="portfolio_return")


def ccc_portfolio_vol(vol_df:        pd.DataFrame,
                      corr_constant: np.ndarray,
                      weights:       dict,
                      tickers:       list = TICKERS) -> pd.Series:
    """
    Constant Conditional Correlation portfolio volatility.
    H_t = D_t * R_bar * D_t, where R_bar never changes over time
    — only the diagonal (individual vols) varies day-to-day.
    Returns DAILY (not annualised) portfolio vol — what VaR needs.
    """
    w = np.array([weights[t] for t in tickers])
    n = len(vol_df)
    port_vol = np.zeros(n)

    for i in range(n):
        d_daily = vol_df.iloc[i][tickers].values / np.sqrt(252)
        H = np.outer(d_daily, d_daily) * corr_constant
        port_vol[i] = np.sqrt(max(w @ H @ w, 1e-12))

    return pd.Series(port_vol, index=vol_df.index, name="port_vol_ccc")


def dcc_portfolio_vol(vol_df:  pd.DataFrame,
                      R_list:  np.ndarray,
                      dates:   pd.DatetimeIndex,
                      weights: dict,
                      tickers: list = TICKERS) -> pd.Series:
    """
    Dynamic Conditional Correlation portfolio volatility.
    Same formula as CCC but R_t comes from the DCC recursion
    and changes every day — correlation breakdowns during
    stress periods are captured here.
    """
    w = np.array([weights[t] for t in tickers])
    aligned_vol = vol_df.loc[dates]
    port_vol = np.zeros(len(dates))

    for i in range(len(dates)):
        d_daily = aligned_vol.iloc[i][tickers].values / np.sqrt(252)
        H = np.outer(d_daily, d_daily) * R_list[i]
        port_vol[i] = np.sqrt(max(w @ H @ w, 1e-12))

    return pd.Series(port_vol, index=dates, name="port_vol_dcc")


def compute_var_cvar(daily_vol: np.ndarray,
                     alpha:      float) -> tuple:
    """
    Gaussian parametric VaR / CVaR — same formula used for
    single-asset risk in Day 10, now applied to portfolio vol.
    """
    z   = stats.norm.ppf(alpha)
    phi = stats.norm.pdf(z)
    VaR  = z * daily_vol
    CVaR = -phi / alpha * daily_vol
    return VaR, CVaR


def kupiec_pof_test(violations: np.ndarray,
                    alpha:      float) -> dict:
    """
    Kupiec POF test — same logic as Day 10's single-asset version,
    applied here to portfolio-level VaR violations.
    """
    n = len(violations)
    x = int(violations.sum())

    if x == 0 or x == n:
        return {"p_value": 1.0, "well_calibrated": True,
                "observed_rate": x / n}

    pi_hat  = x / n
    ll_null = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
    ll_alt  = (n - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)
    LR      = -2 * (ll_null - ll_alt)
    p_value = 1 - stats.chi2.cdf(LR, df=1)

    return {"p_value": round(float(p_value), 4),
           "well_calibrated": bool(p_value >= 0.05),
           "observed_rate": round(pi_hat, 4)}


def christoffersen_independence_test(violations: np.ndarray) -> dict:
    """Same independence test as Day 10, reused at portfolio level."""
    v = violations.astype(int)
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(v)):
        if v[i-1] == 0 and v[i] == 0: n00 += 1
        elif v[i-1] == 0 and v[i] == 1: n01 += 1
        elif v[i-1] == 1 and v[i] == 0: n10 += 1
        elif v[i-1] == 1 and v[i] == 1: n11 += 1

    n0, n1 = n00 + n01, n10 + n11
    if n0 == 0 or n1 == 0 or n01 == 0 or n11 == 0:
        return {"p_value": np.nan, "independent": True}

    pi01 = n01 / n0
    pi11 = n11 / n1
    pi   = (n01 + n11) / (n0 + n1)

    ll_null = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    ll_alt  = (n00 * np.log(1 - pi01) + n01 * np.log(pi01) +
              n10 * np.log(1 - pi11) + n11 * np.log(pi11))
    LR = -2 * (ll_null - ll_alt)
    p_value = 1 - stats.chi2.cdf(LR, df=1)

    return {"p_value": round(float(p_value), 4),
           "independent": bool(p_value >= 0.05)}


def run_portfolio_var_analysis(cov_results: dict = None,
                                dcc_results: dict = None) -> dict:
    """
    Full pipeline: get covariance + DCC results (computing them
    if not already provided) → portfolio returns → CCC vs DCC
    volatility forecasts → VaR/CVaR → backtest both.
    """
    print(f"\n{'='*55}")
    print("  DAY 11 — Portfolio VaR/CVaR: CCC vs DCC")
    print(f"{'='*55}")

    if cov_results is None:
        cov_results = run_realized_covariance()
    if dcc_results is None:
        dcc_results = run_dcc_analysis()

    returns = cov_results["returns"]
    vol_df  = dcc_results["vol_df"]
    dates   = dcc_results["dates"]
    R_list  = dcc_results["R_list"]

    port_ret = portfolio_returns(returns, WEIGHTS)

    full_corr = returns.corr().values
    print("\n  Static (full-sample) correlation matrix:")
    print(pd.DataFrame(full_corr, index=TICKERS,
                       columns=TICKERS).round(4).to_string())

    port_vol_ccc = ccc_portfolio_vol(vol_df, full_corr, WEIGHTS)
    port_vol_dcc = dcc_portfolio_vol(vol_df, R_list, dates, WEIGHTS)

    common_idx = port_vol_dcc.index.intersection(port_ret.index)
    common_idx = common_idx[-TEST_SIZE:]

    actual       = port_ret.loc[common_idx].values
    vol_ccc_test = port_vol_ccc.loc[common_idx].values
    vol_dcc_test = port_vol_dcc.loc[common_idx].values

    print(f"\n  Backtesting on last {len(common_idx)} days...")

    summary_rows = []
    series_store = {}

    for label, vol_test in [("CCC", vol_ccc_test), ("DCC", vol_dcc_test)]:
        for alpha in ALPHAS:
            VaR, CVaR  = compute_var_cvar(vol_test, alpha)
            violations = (actual < VaR).astype(int)
            kup        = kupiec_pof_test(violations, alpha)
            chr_       = christoffersen_independence_test(violations)

            summary_rows.append({
                "Method"          : label,
                "Confidence"      : f"{int((1-alpha)*100)}%",
                "N"               : len(violations),
                "N_violations"    : int(violations.sum()),
                "Observed_rate_%" : round(violations.mean()*100, 2),
                "Expected_rate_%" : round(alpha*100, 2),
                "Kupiec_p"        : kup["p_value"],
                "Calibrated"      : kup["well_calibrated"],
                "Christoffersen_p": chr_["p_value"],
                "Independent"     : chr_["independent"],
                "Mean_VaR_%"      : round(VaR.mean()*100, 3),
                "Mean_CVaR_%"     : round(CVaR.mean()*100, 3),
            })

            series_store[(label, alpha)] = {
                "dates": common_idx, "actual": actual,
                "VaR": VaR, "CVaR": CVaR, "violations": violations,
            }

            status = "CALIBRATED" if kup["well_calibrated"] else "MISCALIBRATED"
            print(f"    {label} {int((1-alpha)*100)}%: observed={violations.mean()*100:.2f}%  expected={alpha*100:.0f}%  {status}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(OUT_DIR, "day11_portfolio_var_backtest.csv"),
        index=False
    )

    # Diversification benefit: weighted-sum-of-vols vs actual portfolio vol
    indiv_daily_vol  = vol_df[TICKERS] / np.sqrt(252)
    weighted_sum_vol = sum(WEIGHTS[t] * indiv_daily_vol[t] for t in TICKERS)

    div_df = pd.DataFrame({
        "weighted_sum_vol"  : weighted_sum_vol.loc[dates],
        "portfolio_vol_dcc" : port_vol_dcc,
        "portfolio_vol_ccc" : port_vol_ccc.loc[dates],
    })
    div_df["diversification_benefit"] = (
        div_df["weighted_sum_vol"] - div_df["portfolio_vol_dcc"]
    )
    div_df.to_csv(os.path.join(OUT_DIR, "day11_diversification_benefit.csv"))

    print(f"\n  ✓ day11_portfolio_var_backtest.csv")
    print(f"  ✓ day11_diversification_benefit.csv")
    print(f"\n  Mean diversification benefit: "
          f"{div_df['diversification_benefit'].mean()*100:.4f}% daily vol")

    return {
        "summary_df"  : summary_df,
        "series_store": series_store,
        "div_df"      : div_df,
        "port_ret"    : port_ret,
        "full_corr"   : full_corr,
        "port_vol_ccc": port_vol_ccc,
        "port_vol_dcc": port_vol_dcc,
    }


if __name__ == "__main__":
    run_portfolio_var_analysis()