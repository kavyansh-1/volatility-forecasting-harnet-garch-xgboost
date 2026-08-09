import os 
import warnings 
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd 
from statsmodels.tsa.api import VAR 

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS = ["SPY", "QQQ", "AAPL"]
FEVD_HORIZON = 10
ROLL_WIN = 252
ROLL_STEP = 21 

def fevd_from_var(rv_window: pd.DataFrame , n_lags : int , horizon : int )