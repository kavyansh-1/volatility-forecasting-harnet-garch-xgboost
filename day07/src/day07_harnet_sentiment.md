# day07_harnet_sentiment.py — Explainer

## Purpose
Modifies the Convolutional Neural Network (HARNet) to accept an additional dense input branch for NLP sentiment.

## HARNetSentiment Architecture
- Uses the same 1d, 5d, 22d parallel convolutions for the numerical price data (`x_temp`).
- Adds `fc_sentiment` to process the single compound sentiment score (`x_sent`) into an 8-dimensional embedding.
- Concatenates the temporal convolutions and the sentiment embedding together before the final fully connected layers.

## Things That Will Break If You Change X
- If you don't scale the sentiment score appropriately: The neural network will struggle to balance the weights between the convolved price features and the raw sentiment score.
