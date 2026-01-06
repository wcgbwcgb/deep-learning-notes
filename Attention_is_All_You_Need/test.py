import torch
from torch import nn

embedding = nn.Embedding(10, 512)  # 10 token, 512 dimensions

token_ids = torch.tensor([
    [0, 3, 5, 7],
    [2, 4, 6, 1],
    [9, 8, 0, 5]
], dtype=torch.long)  # 3 batch

output = embedding(token_ids)  # shape: [3, 4, 512] => [batch, token, dimension]
print(output.shape)