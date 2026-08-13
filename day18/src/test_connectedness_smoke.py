"""Smoke test for the fixed day18_connectedness module (synthetic data)."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Projects/Comparative Analysis of GARCH and ML Models for Volatility Prediction/day18/src")
import day18_connectedness as dc

rng = np.random.default_rng(42)
# 700 daily obs, 3 assets, correlated realized volatilities
n = 700
base = rng.lognormal(mean=0.0, sigma=0.3, size=(n, 3))
rv = pd.DataFrame(base, columns=dc.TICKERS)
rv += 0.6 * rv.shift(1).fillna(0)   # add serial dependence so VAR is estimable

# 1. FEVD
fevd = dc.fevd_from_var(rv, n_lags=1)
assert fevd.shape == (3, 3), f"FEVD shape {fevd.shape} != (3,3)"
assert np.allclose(fevd.sum(axis=1), 1.0, atol=1e-6), "FEVD rows should sum to 1"
print("FEVD OK, shape", fevd.shape, "row sums ~1:", np.allclose(fevd.sum(axis=1), 1.0, atol=1e-6))

# 2. Diebold-Yilmaz table
tbl = dc.diebold_yilmaz_table(fevd)
print("DY table:\n", tbl.to_string())
assert "TO" in tbl.index and "FROM" in tbl.columns
assert "NET" in tbl.index and "TCI" in tbl.index
assert tbl.loc["NET", "SPY"] == tbl.loc["NET", "SPY"], "NET row intact"
assert not np.isnan(tbl.loc["NET", "SPY"]), "NET row must not be wiped to NaN"

# 3. TCI
tci = dc.total_connectedness_index(fevd)
assert 0.0 <= tci <= 1.0, f"TCI {tci} out of [0,1]"
print("TCI =", round(tci, 4))

# 4. Rolling connectedness
roll = dc.rolling_connectedness(rv, win=252, step=63, n_lags=1)
print("Rolling rows:", len(roll))
assert len(roll) >= 3, "expected multiple rolling windows"
assert set(["TCI", "TO_SPY", "FROM_SPY", "NET_SPY"]).issubset(roll.columns)
assert roll["TCI"].notna().all()
print("Rolling TCI range:", round(roll["TCI"].min(), 4), "-", round(roll["TCI"].max(), 4))
print("\nALL CHECKS PASSED")
