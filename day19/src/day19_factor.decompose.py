import numpy as np 
import pandas as pd 
from day19_factor_beta import beta_feature

BURN_IN = 252
TEST_SIZE = 500 

def decompose_returns(asset_ret: pd.Series , market_ret: pd.Series, beta: pd.Series)-> pd.DataFrame: 
    explained = beta * market_ret
    resid = asset_ret - explained
    return pd.DataFrame({"r_explained": explained, "r_idio": resid})

def factor_rv_split(asset_ret : pd.Series , market_ret: pd.Series, beta: pd.Series)-> pd.DataFrame:
    dec = decompose_returns(asset_ret, market_ret, beta)
    rv_market = dec["r_explained"]**2
    rv_idio = dec["r_idio"]**2
    rv_total =- asset_ret**2
    frac_market = rv_market / rv_total.replace(0,np.nan)
    return pd.DataFrame({
        "rv_total" : rv_total, 
        "rv_market" : rv_market, 
        "rv_idio" : rv_idio,
        "frac_market_explained" : frac_market.clip(0,1)
    })

def build_factor_panel(returns: dict , market_key: str = "SPY")-> dict:
    market_ret = returns["market_key"]
    out = {}
    for tk , ret in returns.items(): 
        if tk == market_key: 
            continue
        beta = beta_feature(ret , market_ret , method = "ewma")
        panel = factor_rv_split(ret ,market_ret , beta)
        panel["beta"] = beta
        out[tk] = panel.dropna()
    return out 