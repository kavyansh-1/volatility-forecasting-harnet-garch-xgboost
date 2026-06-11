# src/compute_volatility.py
# PURPOSE: Compute multiple volatility estimates from OHLCV data.
#
# WHY MULTIPLE ESTIMATORS?
# -------------------------
# "Volatility" isn't directly observable — you have to estimate it.
# Different estimators use different information:
#
#   1. Rolling Close-to-Close (historical vol)
#      Uses only closing prices. Simple but wastes intra-day info.
#      Formula: std(log_returns) * sqrt(252)
#
#   2. Realised Volatility (RV) from daily range
#      Uses High and Low. The range captures intra-day movement that
#      close-to-close misses (a stock that swings 5% but closes flat
#      shows zero return but high volatility).
#      This is what the HAR model is built on.
#
#   3. Parkinson Estimator (1980)
#      A classic range-based estimator. More efficient than close-to-close.
#      Formula: (1 / 4*ln(2)) * (ln(H/L))^2
#      Efficiency gain: ~5x vs close-to-close for the same sample size.
#
#   4. Garman-Klass Estimator (1980)
#      Uses Open, High, Low, Close together. Even more efficient than Parkinson.
#      The most information-dense daily estimator that doesn't need tick data.
#
# Your GARCH model (Week 4) will model rolling close-to-close volatility.
# Your HARNet (Week 6) will use HAR-style RV features.
# Having all four now lets you compare them side by side.

import numpy as np
import pandas as pd
import os


def add_rolling_vol(df: pd.DataFrame,
                    windows: list = [5, 21, 63],
                    return_col: str = "log_return") -> pd.DataFrame:
    """
    Add rolling historical volatility columns (annualised).

    Parameters
    ----------
    df         : DataFrame with a log_return column already computed
    windows    : Rolling window sizes in trading days
                 5 = 1 week, 21 = 1 month, 63 = 1 quarter
    return_col : Name of the return column to use

    Returns
    -------
    df with new columns: rv_rolling_5d, rv_rolling_21d, rv_rolling_63d
    """
    df = df.copy()

    for w in windows:
        col_name = f"rv_rolling_{w}d"

        # std() over a rolling window, then annualise with sqrt(252)
        # min_periods=w ensures we don't get values from partial windows
        df[col_name] = (
            df[return_col]
            .rolling(window=w, min_periods=w)
            .std()
            * np.sqrt(252)
        )

    return df


def add_realized_vol(df: pd.DataFrame,
                     windows: list = [5, 21]) -> pd.DataFrame:
    """
    Add range-based realised volatility estimates.

    Parkinson estimator (daily):
        RV_park = (1 / (4 * ln2)) * (ln(H/L))^2
    This gives a daily variance estimate. Roll it over a window for
    a period estimate, then take the sqrt and annualise.

    Garman-Klass estimator (daily):
        RV_gk = 0.5*(ln(H/L))^2 - (2*ln2 - 1)*(ln(C/O))^2
    Uses open-to-close and high-low jointly. More efficient than Parkinson.
    """
    df = df.copy()

    # ── Daily Parkinson variance ──────────────────────────────────────────────
    log_hl = np.log(df["High"] / df["Low"])
    park_daily = (1 / (4 * np.log(2))) * (log_hl ** 2)
    df["park_daily_var"] = park_daily

    # ── Daily Garman-Klass variance ───────────────────────────────────────────
    log_hl2 = np.log(df["High"] / df["Low"]) ** 2
    log_co2 = np.log(df["Close"] / df["Open"]) ** 2
    gk_daily = 0.5 * log_hl2 - (2 * np.log(2) - 1) * log_co2
    df["gk_daily_var"] = gk_daily

    # ── Roll over windows and annualise ──────────────────────────────────────
    for w in windows:
        # Mean of daily variances over the window, then sqrt and annualise
        df[f"park_vol_{w}d"] = np.sqrt(
            df["park_daily_var"].rolling(w, min_periods=w).mean() * 252
        )
        df[f"gk_vol_{w}d"] = np.sqrt(
            df["gk_daily_var"].rolling(w, min_periods=w).mean() * 252
        )

    return df


def add_all_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function: adds all volatility columns in one call.
    This is what your notebooks will call.
    """
    df = add_rolling_vol(df, windows=[5, 21, 63])
    df = add_realized_vol(df, windows=[5, 21])
    return df


def describe_volatility(dfs: dict) -> pd.DataFrame:
    """
    Summary table of volatility levels across assets.
    """
    rows = []
    for ticker, df in dfs.items():
        rows.append({
            "Ticker"         : ticker,
            "Ann_Vol_21d"    : round(df["rv_rolling_21d"].mean(),  4),
            "Ann_Vol_63d"    : round(df["rv_rolling_63d"].mean(),  4),
            "Parkinson_21d"  : round(df["park_vol_21d"].mean(),    4),
            "GarmanKlass_21d": round(df["gk_vol_21d"].mean(),      4),
            "Max_RV_21d"     : round(df["rv_rolling_21d"].max(),   4),
            "Min_RV_21d"     : round(df["rv_rolling_21d"].min(),   4),
        })
    return pd.DataFrame(rows).set_index("Ticker")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.compute_returns import compute_returns

    tickers = ["SPY", "QQQ", "AAPL"]
    for ticker in tickers:
        df = pd.read_csv(f"data/raw/{ticker}_daily.csv",
                         index_col="Date", parse_dates=True)
        df = compute_returns(df)
        df = add_all_volatility(df)

        print(f"\n{ticker} — volatility columns:")
        vol_cols = [c for c in df.columns if "vol" in c or "rv_" in c]
        print(df[vol_cols].dropna().tail(5).to_string())