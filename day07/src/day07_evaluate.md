# day07_evaluate.py — Explainer

## Purpose
Aggregates the cross-validation and test scores across all 60 days of the project to declare a winner.

## compute_qlike()
- Re-implements the QLIKE mathematical formula natively to evaluate non-neural models (HAR, XGBoost, GARCH) exactly identically to HARNet.

## Things That Will Break If You Change X
- If you use raw MSE to declare a winner: You might select a model that vastly underpredicts volatility, which in a live trading firm would lead to disastrous unhedged risk exposure.
