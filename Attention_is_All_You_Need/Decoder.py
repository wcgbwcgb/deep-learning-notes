from torch import nn
from DecoderLayer import MyDecoderLayer

class MyDecoder(nn.Module):
    def __init__(self, N=6, d_model=512, num_heads=8, d_ff=2048):
        super().__init__()
        self.layers = nn.ModuleList([MyDecoderLayer(d_model, num_heads, d_ff) for i in range(N)]) # Module list Nx6

    def forward(self, x, encoder_input):
        for layer in self.layers:
            x = layer(x, encoder_input)
        return x