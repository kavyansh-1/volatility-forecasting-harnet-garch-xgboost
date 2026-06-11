from pathlib import Path
import pandas as pd

repo = Path(__file__).resolve().parents[1]
# Load processed data if present; otherwise load raw and compute returns/vol
try:
    dfs = {}
    for t in ['SPY','QQQ','AAPL']:
        p = repo / 'data' / 'processed' / f"{t}_processed.csv"
        if p.exists():
            dfs[t] = pd.read_csv(p, index_col='Date', parse_dates=True)
        else:
            dfs[t] = pd.read_csv(repo / 'data' / 'raw' / f"{t}_daily.csv", index_col='Date', parse_dates=True)

    # Import functions locally (avoid package import issues)
    import importlib.util
    def load(path):
        spec = importlib.util.spec_from_file_location(path.stem, str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    cr = load(repo / '.vscode' / 'src' / 'compute_returns.py')
    cv = load(repo / '.vscode' / 'src' / 'compute_volatility.py')
    pr = load(repo / '.vscode' / 'src' / 'plot_returns.py')

    # Ensure returns + vol
    dfs = cr.compute_all_returns(dfs)
    dfs = {t: cv.add_all_volatility(df) for t, df in dfs.items()}

    pr.plot_return_distributions(dfs)
    pr.plot_qq(dfs)
    pr.plot_rolling_volatility(dfs)
    pr.plot_vol_estimator_comparison(dfs, ticker='SPY')
    print('Day 2 plots generated')
except Exception as e:
    print('Error:', e)
    raise
