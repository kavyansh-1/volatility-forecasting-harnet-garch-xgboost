import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"{d_model} must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, steps, _ = x.shape
        x = x.view(bsz, steps, self.n_heads, self.d_head)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_heads, steps, d_head = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, steps, n_heads * d_head)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = self._merge_heads(context)
        context = self.w_o(context)

        # Average over heads for an interpretable (T, T) map.
        attn_avg = attn.mean(dim=1)
        return context, attn_avg


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_w = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)

        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        return x, attn_w


class VolatilityTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int = 32,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 32)
        self.fc2 = nn.Linear(32, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.input_proj(x)
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.dropout(x)

        attn_weights = []
        for layer in self.layers:
            x, attn_w = layer(x)
            attn_weights.append(attn_w)

        x = self.final_norm(x)
        pooled = x.mean(dim=1)

        out = F.relu(self.fc1(pooled))
        out = self.dropout(out)
        out = self.fc2(out).squeeze(-1)
        return out, attn_weights







    


