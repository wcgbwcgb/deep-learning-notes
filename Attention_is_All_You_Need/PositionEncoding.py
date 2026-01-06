from torch import nn
import torch

class MyPositionEncoding(nn.Module):
    def __init__(self, token_num, d_model):
        super().__init__()
        self.PE = torch.zeros(token_num, d_model, dtype=torch.float32)
        i = torch.arange(0, d_model, 2, dtype=torch.float32)  # step = 2
        div = torch.pow(10000, i/d_model)
        pos = torch.arange(0, token_num, dtype=torch.float32).unsqueeze(1)
        self.PE[:, 0::2] = torch.sin(pos/div)  # start:stop:step
        self.PE[:, 1::2] = torch.cos(pos/div)  

    def forward(self, x):
        x = x + self.PE
        return x
    

posencoding = MyPositionEncoding(10, 512)
result = posencoding.forward(torch.rand(10, 512))
print(result)
print("ok")