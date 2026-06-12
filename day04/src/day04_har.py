"""Trains HAR Ridge regression."""
import os, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

if __name__ == "__main__":
    os.makedirs("day04/output", exist_ok=True)
    results = []
    for t in ["SPY", "QQQ", "AAPL"]:
        df = pd.read_csv(f"data/processed/{t}_features.csv", index_col="Date", parse_dates=True)
        X = df[['rv_lag_1', 'rv_lag_5', 'rv_lag_21']]
        y = df['target_rv_1d']
        grid = GridSearchCV(Ridge(), {'alpha': [0.1, 1, 10]}, cv=TimeSeriesSplit(n_splits=5), scoring='neg_root_mean_squared_error')
        grid.fit(X, y)
        results.append({'Ticker': t, 'Best_Alpha': grid.best_params_['alpha'], 'CV_RMSE': -grid.best_score_})
    
    res_df = pd.DataFrame(results)
    res_df.to_csv("day04/output/day04_har_results.csv", index=False)
    print("Saved HAR Baseline results to day04/output/day04_har_results.csv")
    print(res_df)
