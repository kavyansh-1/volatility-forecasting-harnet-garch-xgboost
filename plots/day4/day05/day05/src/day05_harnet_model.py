import torch
import torch.nn as nn
import torch.nn.functional as F


class HARNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        crop = self.conv.padding[0]
        if crop > 0:
            x = x[:, :, :-crop]
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x.squeeze(-1)


class HARNet(nn.Module):
    def __init__(
        self,
        n_features: int = 1,
        seq_len: int = 22,
        n_filters: int = 16,
        fc_hidden: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features

        self.branch_day = HARNetBlock(n_features, n_filters, kernel_size=1)
        self.branch_week = HARNetBlock(n_features, n_filters, kernel_size=5)
        self.branch_month = HARNetBlock(n_features, n_filters, kernel_size=22)

        self.fc_1 = nn.Linear(n_filters * 3, fc_hidden)
        self.dropout = nn.Dropout(p=dropout)
        self.fc_2 = nn.Linear(fc_hidden, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.branch_day(x)
        w = self.branch_week(x)
        m = self.branch_month(x)

        h = torch.cat([d, w, m], dim=1)
        h = F.relu(self.fc_1(h))
        h = self.dropout(h)
        out = self.fc_2(h)
        return out.squeeze(-1)


class QLikeLoss(nn.Module):
    def __init__(self, floor: float = 1e-8):
        super().__init__()
        self.floor = floor

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred, min=self.floor)
        target = torch.clamp(target, min=self.floor)
        return torch.mean(torch.log(pred) + target / pred)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


         









