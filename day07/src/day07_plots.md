# day07_plots.py — Explainer

## Purpose
Generates the concluding visualizations summarizing 60 days of algorithmic development.

## Logic Block
- Creates a bar chart directly comparing the baseline models against their Sentiment-enhanced counterparts. 
- A visual confirmation of whether scraping financial news actually yielded measurable alpha.

## Things That Will Break If You Change X
- If you don't scale the Y-axis properly: Differences of 0.001 in RMSE might look massive or invisible. Set logical `ylim` based on your data distribution.
