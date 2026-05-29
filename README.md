# Volatility Forecasting Project

Comparative analysis of GARCH-style volatility and machine learning workflows for SPY, QQQ, and AAPL.

## Project files

- `EOD_DAY1.py` — downloads daily OHLCV data, runs quality checks, and saves price plots.
- `EOD_DAY2.py` — computes returns, volatility estimators, and diagnostic plots.
- `EOD_DAY3.py` — runs ADF, Ljung-Box, and ACF/PACF diagnostics.

## Data and outputs

- `data/raw/` — downloaded daily CSV files.
- `data/processed/` — enriched datasets with returns and volatility columns.
- `plots/` — generated charts and diagnostics.
- `reports/` — CSV summaries from each day.

## Run order

```bash
python EOD_DAY1.py
python EOD_DAY2.py
python EOD_DAY3.py
```

## Notes

- The original scripts are still kept under `.vscode/`.
- The top-level files are the easiest entry points to run and find in GitHub.