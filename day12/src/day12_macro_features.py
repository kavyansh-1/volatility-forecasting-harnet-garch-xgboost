# ─────────────────────────────────────────────────────────────
# day12_macro_features.py
# Macro and cross-asset regime features derived from
# the RELATIONSHIPS between SPY, QQQ, AAPL returns.
#
# WHY MACRO / REGIME FEATURES?
# ─────────────────────────────────────────────────────────────
# Individual asset volatility is partly driven by:
#   (a) asset-specific news / earnings / events
#   (b) sector dynamics (tech sector beta for AAPL and QQQ)
#   (c) broad market regime (risk-on vs risk-off)
#
# We can capture (b) and (c) from the cross-asset signals
# available within our own dataset:
#   • SPY-QQQ spread vol: when tech leads/lags the broad market
#   • Cross-asset return correlations (rolling):
#     rising correlation = risk-off / crash environment
#   • Dispersion: how different are SPY/QQQ/AAPL returns?
#     High dispersion = stock-picking regime, lower systemic risk
#     Low dispersion = macro-driven, higher correlation + correlated vol
#   • Beta instability: if AAPL's beta to SPY is changing rapidly,
#     regime shifts are occurring, often with elevated vol
#
# These features are computed WITHOUT look-ahead and only from
# data that would realistically be available at forecasting time.
# ─────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS = ["SPY", "QQQ", "AAPL"]


def load_returns_all() -> pd.DataFrame:
    """Load aligned return matrix for all three tickers."""
    series = {}
    for ticker in TICKERS:
        path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        series[ticker] = df["log_return"]
    return pd.DataFrame(series).dropna(how="any")


def add_macro_features(df:      pd.DataFrame,
                        returns: pd.DataFrame,
                        ticker:  str) -> pd.DataFrame:
    """
    Add cross-asset regime and macro features to one ticker's DataFrame.

    Parameters
    ----------
    df      : single-ticker enriched DataFrame (from Modules 1+2)
    returns : aligned 3-ticker return matrix (all on same DatetimeIndex)
    ticker  : which ticker we are enriching ("SPY", "QQQ", or "AAPL")
    """
    df = df.copy()

    # Align to the common date index
    df = df.join(returns.add_prefix("ret_"), how="left")

    # ── 1. Rolling cross-asset correlation ───────────────────────
    # Rolling 21d correlation between this asset and the other two.
    # Rising correlation = market moving together = macro-driven regime.
    for other in TICKERS:
        if other == ticker:
            continue
        col = f"ret_{other}"
        if col in df.columns and f"ret_{ticker}" in df.columns:
            roll_corr = (
                df[f"ret_{ticker}"].shift(1)
                .rolling(21, min_periods=15)
                .corr(df[col].shift(1))
            )
            df[f"roll_corr_{ticker}_{other}_21d"] = roll_corr

    # ── 2. Market dispersion ─────────────────────────────────────
    # Dispersion = cross-sectional std of returns across all 3 assets.
    # Low dispersion: all assets move together (macro / risk-off).
    # High dispersion: idiosyncratic (earnings, news) driving moves.
    ret_cols = [f"ret_{t}" for t in TICKERS if f"ret_{t}" in df.columns]
    if len(ret_cols) == 3:
        df["cross_asset_dispersion"] = (
            df[ret_cols].shift(1).std(axis=1)
        )
        df["dispersion_5d_avg"] = (
            df["cross_asset_dispersion"].rolling(5, min_periods=3).mean()
        )

    # ── 3. SPY vol as market regime signal ───────────────────────
    # If forecasting QQQ or AAPL, add SPY's rolling vol as a
    # "market regime" indicator — high SPY vol = risk-off environment.
    if "rv_rolling_21d" in df.columns:
        df[f"spy_vol_context"] = df["rv_rolling_21d"].shift(1)

    # If not SPY itself, load SPY vol explicitly
    if ticker != "SPY":
        spy_path = os.path.join(DATA_DIR, "SPY_processed.csv")
        if os.path.exists(spy_path):
            spy = pd.read_csv(spy_path, index_col="Date", parse_dates=True)
            if "rv_rolling_21d" in spy.columns:
                df["spy_vol_context"] = spy["rv_rolling_21d"].shift(1).reindex(df.index)

    # ── 4. Beta to SPY (rolling) ─────────────────────────────────
    # Beta = Cov(ticker, SPY) / Var(SPY) over rolling 21-day window.
    # High beta + rising SPY vol = amplified vol for this asset.
    # Rapidly changing beta = regime instability.
    ret_self = df[f"ret_{ticker}"].shift(1) if f"ret_{ticker}" in df.columns else None
    ret_spy  = df["ret_SPY"].shift(1)       if "ret_SPY" in df.columns       else None

    if ret_self is not None and ret_spy is not None:
        roll_cov = ret_self.rolling(21, min_periods=15).cov(ret_spy)
        roll_var = ret_spy.rolling(21, min_periods=15).var()
        df["rolling_beta_21d"] = roll_cov / (roll_var + 1e-12)

        # Beta instability = rolling std of the beta itself
        df["beta_instability_21d"] = (
            df["rolling_beta_21d"].rolling(21, min_periods=10).std()
        )

    # ── 5. QQQ-SPY spread vol ────────────────────────────────────
    # When tech leads the broad market (positive spread vol),
    # risk appetite is high; when it lags, risk-off.
    if "ret_QQQ" in df.columns and "ret_SPY" in df.columns:
        qqq_spy_spread = df["ret_QQQ"].shift(1) - df["ret_SPY"].shift(1)
        df["qqq_spy_spread_vol_21d"] = (
            qqq_spy_spread.rolling(21, min_periods=10).std()
        )
        df["qqq_spy_spread_21d"] = (
            qqq_spy_spread.rolling(21, min_periods=10).mean()
        )

    # ── 6. Market trend regime flag ──────────────────────────────
    # SPY 50-day MA vs 200-day MA (golden/death cross).
    # 1 = golden cross (bull trend), 0 = death cross (bear trend).
    # Vol is typically lower in bull trends, higher in bear trends.
    if "ret_SPY" in df.columns:
        spy_close = returns["SPY"].cumsum().apply(np.exp)
        ma50  = spy_close.rolling(50,  min_periods=40).mean()
        ma200 = spy_close.rolling(200, min_periods=150).mean()
        bull_flag = (ma50 > ma200).astype(float)
        df["bull_regime_flag"] = bull_flag.shift(1).reindex(df.index)

    # Drop the raw return columns we joined (keep only derived features)
    df = df.drop(columns=[c for c in df.columns
                           if c.startswith("ret_")], errors="ignore")

    return df


def run_macro_features(tech_dfs: dict = None) -> dict:
    """Add macro/regime features for all tickers."""
    print("\n" + "-"*50)
    print("  Macro and Regime Features")
    print("-"*50)

    returns = load_returns_all()
    if returns.empty:
        print("  ⚠ Could not load aligned returns — skipping macro features")
        return tech_dfs or {}

    results = {}
    for ticker in TICKERS:
        if tech_dfs and ticker in tech_dfs:
            df = tech_dfs[ticker]
        else:
            path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, index_col="Date", parse_dates=True)

        df = add_macro_features(df, returns, ticker)

        macro_cols = [c for c in df.columns if any(k in c for k in
                     ["roll_corr", "dispersion", "spy_vol",
                      "beta", "qqq_spy", "bull_regime"])]
        print(f"  {ticker}: {len(macro_cols)} macro features added")
        results[ticker] = df

    return results