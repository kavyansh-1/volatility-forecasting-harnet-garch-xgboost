import numpy as np 
import pandas as pd 


SEQ_LEN = 22
EWMA_LAMBDA_COV = 0.94 # fixed project constant 

def rolling_ols_beta(asset_ret: pd.Series , market_ret : pd.Series , window : int = SEQ_LEN)->pd.Series: 
    cov = asset_ret.rolling(window).cov(market_ret)
    var = market_ret.rolling(window).var()
    beta = cov / var
    return beta 

def ewma_beta(asset_ret:pd.Series , market_ret:pd.Series , lam : float = EWMA_LAMBDA_COV)->pd.Series:
    a = asset_ret.values
    m = market_ret.values
    n = len(a)
    cov = np.full(n, np.nan)
    var = np.full(n, np.nan)
    beta = np.full(n, np.nan)
    # seed with first SEQ_LEN obs (simple sample estimates)
    if n <= SEQ_LEN:
        return pd.Series(beta, index=asset_ret.index)
    cov0 = np.cov(a[:SEQ_LEN], m[:SEQ_LEN])[0, 1]
    var0 = np.var(m[:SEQ_LEN])
    cov[SEQ_LEN - 1], var[SEQ_LEN - 1] = cov0, var0
    for t in range(SEQ_LEN, n):
        cov[t] = lam * cov[t - 1] + (1 - lam) * a[t] * m[t]
        var[t] = lam * var[t - 1] + (1 - lam) * m[t] ** 2
        beta[t] = cov[t] / var[t] if var[t] > 0 else np.nan
    return pd.Series(beta, index=asset_ret.index)

def beta_feature(asset_ret: pd.Series, market_ret: pd.Series, method: str = "ewma") -> pd.Series:
    """Returns the leakage-safe (shift(1)) beta feature ready for merging into the
    feature matrix."""
    raw = ewma_beta(asset_ret, market_ret) if method == "ewma" else rolling_ols_beta(asset_ret, market_ret)
    return raw.shift(1)