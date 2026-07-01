import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    def __init__(self, n_features: int, d_model: int = 32, dropout: float = 0.1):
        super().__init__()
        self.scale = d_model ** -0.5
        self.w_q = nn.Linear(n_features, d_model, bias=False)
        self.w_k = nn.Linear(n_features, d_model, bias=False)
        self.w_v = nn.Linear(n_features, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(d_model, 32)
        self.fc2 = nn.Linear(32, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.bmm(attn, v)
        context = self.out_proj(context)
        context = self.layer_norm(context + q)

        pooled = context.mean(dim=1)
        out = F.relu(self.fc1(pooled))
        out = self.dropout(out)
        out = self.fc2(out).squeeze(-1)

        return out, attn


class TemporalAttentionWithPositional(TemporalAttention):
    def __init__(self, n_features: int, d_model: int = 32, seq_len: int = 32, dropout: float = 0.1):
        super().__init__(n_features=n_features, d_model=d_model, dropout=dropout)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, n_features) * 0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.pos_embed[:, : x.size(1), :]
        return super().forward(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)











        

