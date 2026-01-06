from torch import nn
import torch

class MyScaledProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        similarity = torch.matmul(query, key.transpose(-1, -2))
        scaled_similarity =  similarity / torch.sqrt(key.size(-1)) 
        softmax = self.fn(scaled_similarity)
        output = torch.matmul(softmax, value)
        return output