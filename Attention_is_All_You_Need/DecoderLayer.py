from torch import nn
from MultiHeadAttention import MyMultiHeadAttention

class MyDecoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.attn1 = MyMultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.attn2 = MyMultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        attn_output1 = self.attn1(x)
        x = self.norm1(x + attn_output1)
        attn_output2 = self.attn2(x)
        x = self.norm2(x + attn_output2)
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)
        return x