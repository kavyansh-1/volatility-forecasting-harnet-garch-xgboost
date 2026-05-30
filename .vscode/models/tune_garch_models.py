warnings.filterwarnings("ignore")
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

try:
    from arch import arch_model

    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("WARNING arch library not found, kindly install with pip install arch")


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    vol_model: str = "GARCH",
    dist: str = "t",
    last_n: int = 1500,
) -> dict:
    if not ARCH_AVAILABLE:
        return {
            "p": p,
            "q": q,
            "vol_model": vol_model,
            "dist": dist,
            "aic": np.nan,
            "bic": np.nan,
            "loglik": np.nan,
            "alpha_sum": np.nan,
            "beta_sum": np.nan,
            "persistence": np.nan,
            "omega": np.nan,
            "converged": False,
            "error": "arch not installed",
        }

    r = returns.dropna().iloc[-last_n:] * 100

    try:
        am = arch_model(r, p=p, q=q, vol=vol_model, dist=dist, rescale=False)
        result = am.fit(disp="off", show_warning=False)

        params = result.params.to_dict()
        alpha_sum = sum(v for k, v in params.items() if "alpha" in k.lower())
        beta_sum = sum(v for k, v in params.items() if "beta" in k.lower())

        return {
            "p": p,
            "q": q,
            "vol_model": vol_model,
            "dist": dist,
            "aic": round(result.aic, 4),
            "bic": round(result.bic, 4),
            "loglik": round(result.loglikelihood, 4),
            "alpha_sum": round(alpha_sum, 6),
            "beta_sum": round(beta_sum, 6),
            "persistence": round(alpha_sum + beta_sum, 6),
            "omega": params.get("omega", np.nan),
            "converged": result.convergence_flag == 0,
            "_result": result,
        }

    except Exception as exc:
        return {
            "p": p,
            "q": q,
            "vol_model": vol_model,
            "dist": dist,
            "aic": np.nan,
            "bic": np.nan,
            "loglik": np.nan,
            "alpha_sum": np.nan,
            "beta_sum": np.nan,
            "persistence": np.nan,
            "omega": np.nan,
            "converged": False,
            "error": str(exc),
        }


def tune_garch(
    df: pd.DataFrame,
    ticker: str,
    p_values: list = [1, 2],
    q_values: list = [1, 2],
    vol_models: list = ["GARCH", "EGARCH"],
    dists: list = ["normal", "t", "skewt"],
) -> dict:
    print(f"Tuning GARCH for {ticker}...")
    returns = df["log_return"]
    rows = []

    for vol in vol_models:
        for dist in dists:
            for p in p_values:
                for q in q_values:
                    result_row = fit_garch(returns, p=p, q=q, vol_model=vol, dist=dist)
                    result_row["ticker"] = ticker
                    rows.append(result_row)

    grid_df = pd.DataFrame(rows)
    grid_df = grid_df[grid_df["aic"].notna()].copy()
    grid_df = grid_df.sort_values("aic").reset_index(drop=True)

    if len(grid_df) == 0:
        print(f"All GARCH fits failed for {ticker}")
        return {"ticker": ticker, "grid_df": grid_df, "best_row": None}

    best_row = grid_df.iloc[0]
    best_result = best_row.get("_result", None)

    print(
        f"Best: {best_row['vol_model']}({best_row['p']}, {best_row['q']})"
        f"-{best_row['dist']} AIC={best_row['aic']:.2f}, "
        f"persistence={best_row.get('persistence', 'n/a')}"
    )

    return {
        "ticker": ticker,
        "grid_df": grid_df.drop(columns=["_result"], errors="ignore"),
        "best_row": best_row,
        "best_result": best_result,
    }


def tune_garch_all_tickers(dfs: dict, **kwargs) -> dict:
    print("\n--- GARCH MODEL TUNING ---")
    results = {}
    for ticker, df in dfs.items():
        results[ticker] = tune_garch(df, ticker, **kwargs)
    return results


if __name__ == "__main__":
    from models.feature_engineering import load_ticker_frames

    tickers = ["SPY", "QQQ", "AAPL"]
    dfs = load_ticker_frames(tickers)
    results = tune_garch_all_tickers(dfs)
    for ticker, result in results.items():
        if result["best_row"] is not None:
            best_row = result["best_row"]
            print(f"{ticker}: best AIC={best_row['aic']:.2f}")
        else:
            print(f"{ticker}: no successful GARCH fit")
    
    
