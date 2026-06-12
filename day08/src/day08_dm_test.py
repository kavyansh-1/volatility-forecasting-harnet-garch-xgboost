# ─────────────────────────────────────────────────────────────
# day08_dm_test.py
# Full Diebold-Mariano (1995) test implementation.
# Tests whether two forecasting models have statistically
# different predictive accuracy.
#
# WHAT IS THE DM TEST?
# ─────────────────────────────────────────────────────────────
# Given two forecasts yhat1 and yhat2 for the same target y:
#   d_t = L(e1_t) - L(e2_t)
#   where L() is a loss function (MSE or absolute error)
#   e1_t = y_t - yhat1_t
#   e2_t = y_t - yhat2_t
#
# H0: E[d_t] = 0  (equal predictive accuracy)
# H1: E[d_t] != 0 (one model is better)
#
# DM statistic: S1 = d_bar / sqrt( (1/T) * f_0 )
# where f_0 is a HAC (Newey-West) estimate of the
# spectral density at frequency zero — this corrects
# for autocorrelation in the loss differential series.
#
# Harvey, Leybourne, Newbold (1997) small-sample correction:
#   S1* = S1 * sqrt( (T + 1 - 2h + h*(h-1)/T) / T )
# where h = forecast horizon (1 here).
# ─────────────────────────────────────────────────────────────

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_hac

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)


def newey_west_var(d: np.ndarray,
                   max_lags: int = None) -> float:
    """
    Newey-West heteroskedasticity and autocorrelation consistent
    (HAC) variance estimator for the loss differential series d_t.

    Formula:
        Var_NW = gamma_0 + 2 * sum_{k=1}^{L} w_k * gamma_k
        w_k    = 1 - k/(L+1)   (Bartlett kernel weights)
        gamma_k = (1/T) * sum_{t=k+1}^{T} d_t * d_{t-k}

    WHY HAC?
    The loss differential d_t is often autocorrelated when the
    forecast error is autocorrelated (which it almost always is
    in volatility forecasting). Using simple var(d)/T would give
    a downward-biased standard error and over-reject H0.
    HAC corrects for this by incorporating autocovariances.

    Parameters
    ----------
    d        : loss differential array (e1^2 - e2^2)
    max_lags : number of lags for HAC. Default: floor(T^(1/3))
               — Andrews (1991) rule of thumb for financial data.
    """
    T = len(d)
    if max_lags is None:
        max_lags = int(np.floor(T ** (1 / 3)))

    d_dm    = d - d.mean()   # demean
    gamma_0 = np.mean(d_dm ** 2)

    hac_var = gamma_0
    for k in range(1, max_lags + 1):
        weight  = 1 - k / (max_lags + 1)          # Bartlett weight
        gamma_k = np.mean(d_dm[k:] * d_dm[:-k])   # autocovariance at lag k
        hac_var += 2 * weight * gamma_k

    return max(hac_var, 1e-12)   # floor to avoid division by zero


def dm_test(y:     np.ndarray,
             yhat1: np.ndarray,
             yhat2: np.ndarray,
             h:     int   = 1,
             loss:  str   = "mse",
             label1: str  = "Model1",
             label2: str  = "Model2") -> dict:
    """
    Full Diebold-Mariano test with Harvey-Leybourne-Newbold
    small-sample correction.

    Parameters
    ----------
    y      : actual values
    yhat1  : forecasts from model 1 (baseline / reference)
    yhat2  : forecasts from model 2 (challenger)
    h      : forecast horizon in steps (1 for one-step-ahead)
    loss   : 'mse' (squared error) or 'mae' (absolute error)
    label1 : name for model 1 in output dict
    label2 : name for model 2 in output dict

    Returns
    -------
    dict with keys:
        DM_stat     : Harvey-Leybourne-Newbold corrected statistic
        p_value     : two-sided p-value under t(T-1) distribution
        significant : bool — reject H0 at 5%?
        better_model: which model has lower average loss
        mean_d      : mean loss differential (positive = model2 better)
        label1, label2
    """
    y, yhat1, yhat2 = map(np.asarray, [y, yhat1, yhat2])

    # Compute per-period losses
    if loss == "mse":
        L1 = (y - yhat1) ** 2
        L2 = (y - yhat2) ** 2
    elif loss == "mae":
        L1 = np.abs(y - yhat1)
        L2 = np.abs(y - yhat2)
    else:
        raise ValueError(f"loss must be 'mse' or 'mae', got {loss}")

    # Loss differential: positive = model2 has smaller loss
    d = L1 - L2
    T = len(d)

    # HAC variance
    hac_var = newey_west_var(d)

    # Raw DM statistic
    dm_raw  = d.mean() / np.sqrt(hac_var / T)

    # Harvey-Leybourne-Newbold small-sample correction
    # Adjusts for the fact that short samples inflate rejection rates
    correction = np.sqrt(
        (T + 1 - 2 * h + h * (h - 1) / T) / T
    )
    dm_stat = dm_raw * correction

    # Two-sided p-value using t(T-1) distribution
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=T - 1))

    better = label2 if d.mean() > 0 else label1

    return {
        "label1"      : label1,
        "label2"      : label2,
        "loss"        : loss,
        "T"           : T,
        "mean_d"      : round(float(d.mean()),  8),
        "DM_stat"     : round(float(dm_stat),   4),
        "p_value"     : round(float(p_value),   4),
        "significant" : bool(p_value < 0.05),
        "better_model": better,
    }


def run_all_dm_tests(predictions: dict) -> pd.DataFrame:
    """
    Run DM tests for every pair of models within each ticker.
    Also tests baseline vs augmented for each model.

    Parameters
    ----------
    predictions : dict structured as:
        {
          ticker: {
            model_version: np.ndarray of predictions,
            ...
          },
          "y_test": {ticker: np.ndarray}
        }

    Naming convention for model_version keys:
        "HAR_base", "HAR_aug",
        "GARCH_base",
        "XGB_base", "XGB_aug",
        "HARNet_base", "HARNet_aug"

    Returns
    -------
    pd.DataFrame with one row per comparison
    """
    rows = []
    tickers = [t for t in predictions.keys() if t != "y_test"]

    for ticker in tickers:
        y       = predictions["y_test"][ticker]
        models  = predictions[ticker]
        keys    = list(models.keys())

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                if models[k1] is None or models[k2] is None:
                    continue

                result = dm_test(
                    y      = y,
                    yhat1  = models[k1],
                    yhat2  = models[k2],
                    label1 = k1,
                    label2 = k2,
                    loss   = "mse",
                )
                result["ticker"] = ticker
                rows.append(result)

    dm_df = pd.DataFrame(rows)
    if not dm_df.empty:
        out = os.path.join(OUT_DIR, "day08_dm_results.csv")
        dm_df.to_csv(out, index=False)
        print(f"  ✓ DM results → {out}")

    return dm_df


def summarise_dm(dm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a win/loss matrix:
    For each model, how many times did it significantly
    outperform another model across all tickers?

    Returns a DataFrame where entry [i,j] = number of tickers
    where model i significantly beat model j (p < 0.05).
    """
    if dm_df.empty:
        return pd.DataFrame()

    models = sorted(
        set(dm_df["label1"].tolist() + dm_df["label2"].tolist())
    )
    matrix = pd.DataFrame(0, index=models, columns=models)

    for _, row in dm_df.iterrows():
        if row["significant"]:
            matrix.loc[row["better_model"], row["label1"]] += 1
            matrix.loc[row["better_model"], row["label2"]] += 1

    out = os.path.join(OUT_DIR, "day08_dm_win_matrix.csv")
    matrix.to_csv(out)
    print(f"  ✓ DM win matrix → {out}")
    return matrix