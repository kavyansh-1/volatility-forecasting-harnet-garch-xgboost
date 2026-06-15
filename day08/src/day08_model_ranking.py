import os 
import warnings 
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd 

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(BASE_DIR, "output")

WEIGHTS = {
    "RMSE" : 0.35,
    "QLIKE" : 0.30,
    "DirAcc" : 0.20, 
    "MAE": 0.15,

}

LOWER_IS_BETTER = ["RMSE" , "MAE" , "QLIKE"]
GREATER_IS_BETTER = ["DirAcc"]

def load_all_metrics() -> pd.DataFrame:

    search_paths = [
        os.path.join(BASE_DIR, "..", "day04", "output", "day04_metrics.csv"),
        os.path.join(BASE_DIR, "..", "day05", "output",
                     "day05_all_model_metrics.csv"),
        os.path.join(BASE_DIR, "..", "day07", "output",
                     "day07_all_metrics.csv"),
    ]

    dfs = []
    for path in search_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"  ✓ Loaded: {os.path.basename(path)}"
                  f"  ({len(df)} rows)")
        else:
            print(f" Not found: {os.path.basename(path)}"
                  f" — skipping this.. ")

    if not dfs:
        print("  ⚠ No metrics CSVs found. Generating synthetic data.")
        return _generate_synthetic_metrics()

    combined = pd.concat(dfs, ignore_index=True)
    
    combined.columns = [c.strip() for c in combined.columns]


    if "Version" in combined.columns:
        mask = combined["Version"].notna()
        combined.loc[mask, "Model"] = (
            combined.loc[mask, "Model"] + "_" +
            combined.loc[mask, "Version"]
        )
        combined = combined.drop(columns=["Version"])

    return combined


def _generate_synthetic_metrics() -> pd.DataFrame:
    np.random.seed(42)
    tickers = ["SPY", "QQQ", "AAPL"]
    models  = [
        "HAR_Baseline", "HAR_Augmented",
        "GARCH",
        "XGBoost_Baseline", "XGBoost_Augmented",
        "HARNet_Baseline", "HARNet_Augmented",
    ]

    rows = []
    base_rmse = {"SPY":0.036 , "QQQ": 0.058 , "AAPL": 0.086}
    base_dirAcc = 49.0

    for ticker in tickers:
        for i , model in enumerate(models):
            noise = np.random.uniform(-0.003,0.003)
            aug_boost = -0.001 if "Augmented" in model else 0
            rows.append({
                "Ticker" : ticker , 
                "Model" : model, 
                "RMSE" : round(base_rmse[ticker] + noise + aug_boost , 6),
                "MAE" : round(base_rmse[ticker] * 0.68 + noise , 6),
                "QLIKE" : round(-2.5 + i * 0.1 + noise * 10 , 4),
                "DirAcc" : round(base_dirAcc + np.random.uniform(-3,5),2),
                "N" : 500,


            })
    return pd.DataFrame(rows)

def compute_per_ticker_ranks(df: pd.DataFrame) -> pd.DataFrame:
    all_rows = []

    for ticker in df["Ticker"].unique():
        sub = df[df["Ticker"] == ticker].copy()

        for metric , weight in WEIGHTS.items():
            if metric not in sub.columns:
                continue
            ascending = metric in LOWER_IS_BETTER
            sub[f"rank_{metric}"] = sub[metric].rank(ascending = ascending , method = "min")

        rank_cols = [f"rank_{m}" for m in WEIGHTS if f"rank_{m}" in sub.columns]
        weights_list = [WEIGHTS[m] for m in WEIGHTS if f"rank_{m}" in sub.columns]

        sub["composite_score"] = sum(w * sub[rc] for w , rc in zip(weights_list , rank_cols))

        sub["overall_rank"] = sub["composite_score"].rank(
            method = "min" , ascending = True
        ).astype(int)

        all_rows.append(sub)
        
    return pd.concat(all_rows , ignore_index = True)

def compute_cross_ticker_ranking(df:pd.DataFrame)-> pd.DataFrame:
    agg_cols = ["RMSE" , "MAE" , "QLIKE" , "DirAcc" , "composite_score"]
    agg_cols = [c for c in agg_cols if c in df.columns]

    cross = (
        df.groupby("Model")[agg_cols]
        .mean()
        .reset_index()
        .sort_values("composite_score")
        .reset_index(drop = True)
    )
    cross["final_rank"] = range(1, len(cross) + 1)

    return cross

def run_model_ranking() -> tuple:

    print(f"\n{'='*55}")
    print("  DAY 8 — Model Ranking")
    print(f"{'='*55}")

    raw_df = load_all_metrics()
    ranked_df = compute_per_ticker_ranks(raw_df)
    final_df = compute_cross_ticker_ranking(ranked_df)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    ranked_df.to_csv(
        os.path.join(OUT_DIR , "day08_per_ticker_ranks.csv"), 
        index = False
    )

    final_df.to_csv(
        os.path.join(OUT_DIR , "day08_final_ranking.csv"),
        index = False
    )

    print("\n Final Model Ranking (cross-ticker average):")
    print(final_df[["final_rank" , "Model" , "RMSE" , "DirAcc" , "composite_score"]].to_string(index = False))

    print(f"\n   day08_per_ticker_ranks.csv")
    print(f"  day08_final_ranking.csv")

    return ranked_df , final_df
