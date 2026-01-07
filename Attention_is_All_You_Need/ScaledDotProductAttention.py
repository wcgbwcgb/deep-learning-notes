from torch import nn
import torch
import math

class MyScaledProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        similarity = torch.matmul(query, key.transpose(-1, -2))
        scaled_similarity =  similarity / math.sqrt(key.size(-1)) 

        if mask is not None:
            scaled_similarity = scaled_similarity.masked_fill(mask == True, float('-inf')) # change the masked token into -inf

        softmax = self.fn(scaled_similarity)
        output = torch.matmul(softmax, value)
        return output