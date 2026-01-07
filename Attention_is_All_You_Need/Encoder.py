from torch import nn
from EncoderLayer import MyEncoderLayer

class MyEncoder(nn.Module):
    def __init__(self, N=6, d_model=512, num_heads=8, d_ff=2048):
        super().__init__()
        self.layers = nn.ModuleList([MyEncoderLayer(d_model, num_heads, d_ff) for i in range(N)]) # Module list Nx6

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x