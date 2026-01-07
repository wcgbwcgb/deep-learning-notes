from torch import nn
from MultiHeadAttention import MyMultiHeadAttention
import torch

class MyDecoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, p_dropout=0.1):
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
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x, encoder_input):
        batch_size, seq_len, _ = x.size()
        
        # masked multi head attention
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)

        attn_output1 = self.attn1(x, x, x, mask)
        attn_output1 = self.dropout(attn_output1)
        x = self.norm1(x + attn_output1)

        attn_output2 = self.attn2(x, encoder_input, encoder_input)
        attn_output2 = self.dropout(attn_output2)
        x = self.norm2(x + attn_output2)

        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        x = self.norm3(x + ffn_output)
        return x