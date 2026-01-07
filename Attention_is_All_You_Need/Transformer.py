from torch import nn
from PositionEncoding import MyPositionEncoding
from Encoder import MyEncoder
from Decoder import MyDecoder

class MyTransformer(nn.Module):
    def __init__(self, token_num, d_model=512, num_heads=8, d_ff=2048, N=6, p_dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(token_num, d_model)
        self.pos = MyPositionEncoding(token_num=token_num, d_model=d_model)
        self.encoder = MyEncoder(N=N, d_model=d_model, num_heads=num_heads, d_ff=d_ff)
        self.decoder = MyDecoder(N=N, d_model=d_model, num_heads=num_heads, d_ff=d_ff)
        self.linear = nn.Linear(d_model, token_num)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x, target):
        # embedding
        x = self.embed(x)
        target = self.embed(target)
        # position encoding
        x = self.pos(x)
        target = self.pos(target)
        # dropout
        x = self.dropout(x)
        target = self.dropout(target)
        # encoder and decoder
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(target, encoder_output)
        # linear
        result = self.linear(decoder_output)
        return result