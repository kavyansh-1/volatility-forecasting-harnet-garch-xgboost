import os 
try:
    import yfinance as yf
except Exception as e:
    raise ImportError("yfinance is required. Install with: pip install yfinance") from e

import os
from typing import List, Dict
from datetime import datetime

try:
    import yfinance as yf
except Exception as e:
    raise ImportError("yfinance is required. Install with: pip install yfinance") from e

try:
    import pandas as pd
except Exception as e:
    raise ImportError("pandas is required. Install with: pip install pandas") from e
 

def download_ohlcv(tickers: List[str], start: str, end: str, save_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
    """Download OHLCV daily data for `tickers` from `start` to `end` and save CSVs to `save_dir`.

    Returns a dict mapping ticker -> DataFrame.
    """
    os.makedirs(save_dir, exist_ok=True)
    results: Dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        print(f"Downloading {ticker}...")

        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

        if df is None or df.empty:
            print(f"  Warning: no data for {ticker}")
            results[ticker] = pd.DataFrame()
            continue

        # Ensure index name and flatten multi-level columns before saving
        df.index.name = "Date"
        if getattr(df.columns, "nlevels", 1) > 1:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        path = os.path.join(save_dir, f"{ticker}_daily.csv")
        df.to_csv(path)

        try:
            date_min = df.index.min().date()
            date_max = df.index.max().date()
        except Exception:
            date_min = df.index.min()
            date_max = df.index.max()

        print(f" Rows: {len(df):,} | Range: {date_min} to {date_max} | Saved: {path}")
        results[ticker] = df

    return results


if __name__ == "__main__":
    TICKERS = ["SPY", "QQQ", "AAPL"]
    START = "2015-01-01"
    END = datetime.today().strftime("%Y-%m-%d")

    dfs = download_ohlcv(TICKERS, START, END)
    print(f"\nDownloaded {len(dfs)} assets successfully")



