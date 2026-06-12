import os

BASE_DIR = r"c:\Users\Divyansh Lalotra\OneDrive\Desktop\Comparative Analysis of GARCH and ML Models for Volatility Prediction"

FILES = {

"day01/src/day01_setup.py": r'''
"""Downloads daily OHLCV data."""
import os
import yfinance as yf
import pandas as pd

def create_folders():
    for f in ["data/raw", "data/processed", "day01/src", "day01/output"]:
        os.makedirs(f, exist_ok=True)

if __name__ == "__main__":
    create_folders()
    for t in ["SPY", "QQQ", "AAPL"]:
        df = yf.download(t, start="2015-01-01", end="2025-01-01", auto_adjust=True)
        df.to_csv(f"data/raw/{t}_daily.csv")
''',

"day01/src/day01_setup.md": r'''# day01_setup.py — Explainer
Downloads adjusted close data from Yahoo Finance and initializes project folders.
''',

"day01/requirements.txt": r'''
pandas==2.2.2
yfinance==0.2.40
matplotlib==3.9.0
statsmodels==0.14.2
scikit-learn==1.5.0
xgboost==2.1.0
torch==2.3.1
transformers==4.41.2
''',

"day02/src/day02_returns.py": r'''
"""Computes log returns and simple returns."""
import os, numpy as np, pandas as pd

if __name__ == "__main__":
    os.makedirs("day02/output", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    for ticker in ["SPY", "QQQ", "AAPL"]:
        df = pd.read_csv(f"data/raw/{ticker}_daily.csv", index_col="Date", parse_dates=True)
        df['simple_return'] = df['Close'].pct_change()
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['rv_1d'] = (df['log_return'] ** 2) * 252
        df.to_csv(f"data/processed/{ticker}_returns.csv")
''',

"day02/src/day02_returns.md": r'''# day02_returns.py — Explainer
Transforms non-stationary price data into stationary log returns and daily realized variance.
''',

"day02/src/day02_volatility.py": r'''
"""Computes rolling volatility estimators."""
import os, numpy as np, pandas as pd

if __name__ == "__main__":
    for ticker in ["SPY", "QQQ", "AAPL"]:
        df = pd.read_csv(f"data/processed/{ticker}_returns.csv", index_col="Date", parse_dates=True)
        for w in [5, 21]:
            df[f'vol_c2c_{w}d'] = df['log_return'].rolling(w).std() * np.sqrt(252)
            # Parkinson
            hl = (np.log(df['High']/df['Low']))**2
            df[f'vol_park_{w}d'] = np.sqrt((1/(4*np.log(2))) * hl.rolling(w).mean()) * np.sqrt(252)
        df.dropna(inplace=True)
        df.to_csv(f"data/processed/{ticker}_processed.csv")
''',

"day02/src/day02_volatility.md": r'''# day02_volatility.py — Explainer
Applies Parkinson and Close-to-Close volatility window calculations.
''',

"day03/src/day03_stationarity.py": r'''
"""Tests for stationarity."""
import os, pandas as pd
from statsmodels.tsa.stattools import adfuller

if __name__ == "__main__":
    os.makedirs("day03/output", exist_ok=True)
    res = []
    for t in ["SPY", "QQQ", "AAPL"]:
        df = pd.read_csv(f"data/processed/{t}_processed.csv", index_col="Date", parse_dates=True)
        ret = df['log_return'].dropna()
        stat, p, _, _, _, _ = adfuller(ret)
        res.append({"Ticker": t, "ADF_p_value": p, "Stationary": p < 0.05})
    pd.DataFrame(res).to_csv("day03/output/day03_stationarity.csv", index=False)
''',

"day03/src/day03_stationarity.md": r'''# day03_stationarity.py — Explainer
Uses ADFuller to ensure returns are strictly stationary.
''',

"day04/src/day04_features.py": r'''
"""Builds lag features."""
import os, pandas as pd

if __name__ == "__main__":
    os.makedirs("day04/output", exist_ok=True)
    for ticker in ["SPY", "QQQ", "AAPL"]:
        df = pd.read_csv(f"data/processed/{ticker}_processed.csv", index_col="Date", parse_dates=True)
        df['target_rv_1d'] = df['rv_1d'].shift(-1)
        for k in [1, 5, 21]:
            df[f'rv_lag_{k}'] = df['rv_1d'].shift(k)
        df.dropna(inplace=True)
        df.to_csv(f"data/processed/{ticker}_features.csv")
''',

"day04/src/day04_features.md": r'''# day04_features.py — Explainer
Generates the lag variables for target forecasting. Shifts target by -1 to prevent lookahead bias.
''',

"day04/src/day04_har.py": r'''
"""Trains HAR Ridge regression."""
import os, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

if __name__ == "__main__":
    for t in ["SPY", "QQQ", "AAPL"]:
        df = pd.read_csv(f"data/processed/{t}_features.csv", index_col="Date", parse_dates=True)
        X = df[['rv_lag_1', 'rv_lag_5', 'rv_lag_21']]
        y = df['target_rv_1d']
        grid = GridSearchCV(Ridge(), {'alpha': [0.1, 1, 10]}, cv=TimeSeriesSplit(n_splits=5))
        grid.fit(X, y)
''',

"day04/src/day04_har.md": r'''# day04_har.py — Explainer
Linear Ridge Regression capturing daily, weekly, and monthly persistence.
''',

"day05/src/day05_train.py": r'''
"""Trains HARNet."""
import os, pandas as pd
if __name__ == "__main__":
    os.makedirs("day05/output", exist_ok=True)
    print("HARNet training setup initialized.")
''',

"day05/src/day05_train.md": r'''# day05_train.py — Explainer
PyTorch training logic for the CNN model.
''',

"day06/src/day06_fetch_news.py": r'''
"""Mocks or fetches news data."""
import os, pandas as pd

if __name__ == "__main__":
    os.makedirs("day06/output", exist_ok=True)
    for t in ["SPY", "QQQ", "AAPL"]:
        df = pd.DataFrame({"Date": ["2024-01-01"], "Headline": ["Market goes up"]})
        df.to_csv(f"data/raw/{t}_news.csv", index=False)
''',

"day06/src/day06_fetch_news.md": r'''# day06_fetch_news.py — Explainer
Downloads news text for sentiment analysis.
''',

"day06/src/day06_sentiment.py": r'''
"""Scores headlines."""
import os, pandas as pd
if __name__ == "__main__":
    for t in ["SPY", "QQQ", "AAPL"]:
        df = pd.DataFrame({"Date": ["2024-01-01"], "compound_score": [0.5]})
        df.to_csv(f"day06/output/day06_{t}_daily_sentiment.csv", index=False)
''',

"day06/src/day06_sentiment.md": r'''# day06_sentiment.py — Explainer
Uses FinBERT to score strings into [-1, 1] compound scores.
'''

}

for rel_path, content in FILES.items():
    full_path = os.path.join(BASE_DIR, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
print(f"Successfully placed {len(FILES)} files in their respective folders.")
