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
    
