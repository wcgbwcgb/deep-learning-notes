from torch import nn
from PositionEncoding import MyPositionEncoding
from EncoderLayer import MyEncoderLayer
from DecoderLayer import MyDecoderLayer

class MyTransformer(nn.Module):
    def __init__(self, token_num, d_model=512, num_heads=8, d_ff=2048):
        super().__init__()
        self.embed = nn.Embedding(token_num, d_model)
        self.pos = MyPositionEncoding(token_num=token_num, d_model=d_model)
        self.encoder = nn.Sequential(
            MyEncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff),
            MyEncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff),
            MyEncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff),
            MyEncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff),
            MyEncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff),
            MyEncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
        )
        self.decoder = nn.Sequential(
            MyDecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff),
            MyDecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff),
            MyDecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff),
            MyDecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff),
            MyDecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff),
            MyDecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
        )
        self.linear = nn.Linear()
        self.softmax = nn.Softmax()

    def forward(self, x):
        
        return x