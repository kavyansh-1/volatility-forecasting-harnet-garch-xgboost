"""
Modifies HARNet to accept sentiment as an auxiliary input feature.
"""
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# A simplified HARNet Block
class HARNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(HARNetBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x.squeeze(-1)

class HARNetSentiment(nn.Module):
    def __init__(self, num_temporal_features):
        super(HARNetSentiment, self).__init__()
        # Temporal branches
        self.branch_daily = HARNetBlock(num_temporal_features, 16, kernel_size=1)
        self.branch_weekly = HARNetBlock(num_temporal_features, 16, kernel_size=5)
        self.branch_monthly = HARNetBlock(num_temporal_features, 16, kernel_size=22)
        
        # Sentiment branch
        self.fc_sentiment = nn.Linear(1, 8)
        self.relu = nn.ReLU()
        
        # Fusion layer (3 * 16) + 8 = 56
        self.fc1 = nn.Linear(48 + 8, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x_temp, x_sent):
        d = self.branch_daily(x_temp[:, :, -1:])
        w = self.branch_weekly(x_temp[:, :, -5:])
        m = self.branch_monthly(x_temp)
        
        s = self.relu(self.fc_sentiment(x_sent))
        
        merged = torch.cat((d, w, m, s), dim=1)
        
        out = self.relu(self.fc1(merged))
        out = self.fc2(out)
        return out

if __name__ == "__main__":
    os.makedirs("day07/output", exist_ok=True)
    print("Initialized HARNet with Sentiment fusion architecture.")
