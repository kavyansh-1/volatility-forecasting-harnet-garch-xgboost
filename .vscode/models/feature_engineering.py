import os
import numpy as np
import pandas as pd
from typing import Tuple


def build_rv_target(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
  """Build next-day realized variance target from `log_return`.

  Returns (rv_1d, target) where `target` is rv_1d shifted -1 (next day).
  """
  rv_1d = df["log_return"] ** 2 * 252
  target = rv_1d.shift(-1)  # next day's target with no leakage
  return rv_1d, target


def build_feature_matrix(
  df: pd.DataFrame,
  rv_lags: list = [1, 2, 3, 5, 10, 21],
  return_lags: list = [1, 2, 3, 5],
  rolling_wins: list = [5, 21, 63],
  include_park: bool = True,
  include_gk: bool = True,
) -> pd.DataFrame:
  feats = pd.DataFrame(index=df.index)

  rv_1d = df["log_return"] ** 2 * 252

  for k in rv_lags:
    feats[f"rv_lag_{k}"] = rv_1d.shift(k)

  for k in return_lags:
    feats[f"ret_lag_{k}"] = df["log_return"].shift(k)

  for w in rolling_wins:
    col = f"rv_rolling_{w}d"
    if col in df.columns:
      feats[col] = df[col].shift(1)

  if include_park:
    for col in ["park_vol_5d", "park_vol_21d"]:
      if col in df.columns:
        feats[col] = df[col].shift(1)

  if include_gk:
    for col in ["gk_vol_5d", "gk_vol_21d"]:
      if col in df.columns:
        feats[col] = df[col].shift(1)

  if "rv_rolling_5d" in df.columns and "rv_rolling_21d" in df.columns:
    feats["rv_ratio_5_21"] = (
      df["rv_rolling_5d"].shift(1) / (df["rv_rolling_21d"].shift(1) + 1e-10)
    )

  feats["rv_ewm_10"] = rv_1d.shift(1).ewm(span=10, min_periods=5).mean()
  feats["abs_ret_lag1"] = df["log_return"].abs().shift(1)

  return feats


def prepare_xy(
  df: pd.DataFrame,
  rv_lags: list = [1, 2, 3, 5, 10, 21],
  return_lags: list = [1, 2, 3, 5],
  rolling_wins: list = [5, 21, 63],
  include_park: bool = True,
  include_gk: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
  """Create feature matrix X, target y and rv_1d series aligned and cleaned (dropna).

  Returns (X_clean, y_clean, rv_clean)
  """
  rv_1d, target = build_rv_target(df)
  X = build_feature_matrix(df, rv_lags, return_lags, rolling_wins, include_park, include_gk)
  combined = pd.concat([X, target.rename("target"), rv_1d.rename("rv_1d")], axis=1).dropna()

  X_clean = combined.drop(columns=["target", "rv_1d"])
  y_clean = combined["target"].copy()
  rv_clean = combined["rv_1d"].copy()

  return X_clean, y_clean, rv_clean


def train_test_split_ts(
  X: pd.DataFrame, y: pd.Series, test_size: int = 500
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
  """Time-series train/test split that keeps temporal order.

  Returns (X_train, X_test, y_train, y_test)
  """
  n = len(X)
  if test_size <= 0 or test_size >= n:
    raise ValueError("`test_size` must be >0 and < len(X)")
  tr_idx = n - test_size
  X_train = X.iloc[:tr_idx]
  X_test = X.iloc[tr_idx:]
  y_train = y.iloc[:tr_idx]
  y_test = y.iloc[tr_idx:]
  return X_train, X_test, y_train, y_test


if __name__ == "__main__":
  # Simple runnable demo: try to load `data/raw/AAPL_daily.csv` from workspace root.
  sample_path = os.path.join(os.getcwd(), "data", "raw", "AAPL_daily.csv")
  if os.path.exists(sample_path):
    df = pd.read_csv(sample_path, parse_dates=["date"]).set_index("date")
    if "close" not in df.columns:
      raise RuntimeError(f"expected `close` column in {sample_path}")
  else:
    # create synthetic data
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=800)
    price = 100 + np.cumsum(np.random.normal(0, 1, size=len(dates)))
    df = pd.DataFrame({"close": price}, index=dates)

  df["log_return"] = np.log(df["close"]).diff()

  X, y, rv = prepare_xy(df)
  print("X shape:", X.shape)
  print("y shape:", y.shape)
  print("rv shape:", rv.shape)

  # quick train-test split
  test_size = min(200, len(X) // 4)
  X_tr, X_te, y_tr, y_te = train_test_split_ts(X, y, test_size=test_size)
  print("X_tr, X_te shapes:", X_tr.shape, X_te.shape)
  print("y_tr, y_te shapes:", y_tr.shape, y_te.shape)




