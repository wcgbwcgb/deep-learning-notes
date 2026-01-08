from torch import nn
from MultiHeadAttention import MyMultiHeadAttention

class MyEncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, p_dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MyMultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x, mask=None):
        attn_output = self.attn(x, x, x, mask=mask)
        attn_output = self.dropout(attn_output)
        x = self.norm1(x + attn_output) # Add & Norm

        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        x = self.norm2(x + ffn_output)
        return x
    

import torch

layer = MyEncoderLayer()
x = torch.randn(2, 5, 512)
out = layer(x)
print(out.shape)