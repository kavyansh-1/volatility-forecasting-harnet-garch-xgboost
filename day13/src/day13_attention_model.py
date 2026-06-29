import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 

class TemporalAttention(nn.Module):
    def __init__ (self , n_features : int , d_model : int = 32, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.scale = d_model **-0.5
        self.W_q = nn.Linear(n_features , d_model , bias = False)
        self.W_k = nn.Linear(n_features , d_model , bias = False)
        self.W_v = nn.Linear(n_features , d_model , bias = False)

        self.out_proj = nn.Linear(d_model , d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Droput(p=dropout)

        self.fc1 = nn.Linear(d_model , 32)
        self.fc2 = nn.Linear(32 , 1)

        self.__init__weights()

    def __init__weights(self):
        for m in self.modules():
            if isinstance(m , nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self , x:torch.Tensor)-> tuple:
        B ,T, F = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = torch.bmm(Q , K.transpose(1,2)) * self_scale
        attn = F.softmax(scores , dim = -1)
        attn = self.dropout(attn)

        context = torch.bmm(attn , V)
        context = self.out_proj(context)

        context = self.layer_norm(context + Q)

        pooled = context.mean(dim = 1)

        out = F.relu(self.fc1(pooled))
        out = self.dropout(out)
        out = self.fc2(out).squeeze(-1)

        return out , attn 

class TemporalAttentionModelWithPositional(TemporalAttention):

    def __init__(self , n_features: int, d_model: int = 32 , seq_len: int = 22 , dropout: float = 0.1):
        super().__init__(n_features , d_model , dropout)
        self.seq_len = seq_len
        self.pos_embed = nn.Parameter(torch.randn(1 , seq_len , n_features)*0.02)

    def forward(self , x: torch.Tensor)-> tuple:
        x = x + self.pos_embed[: , :x.size(1), :]
        return super().forward(x)

def count_parameters(model: nn.Module)-> int:
    return sum(p.nume1() for p in model.parameters() if p.requires_grad)











        

