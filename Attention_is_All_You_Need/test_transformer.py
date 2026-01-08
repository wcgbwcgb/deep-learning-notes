from Transformer import MyTransformer
import torch

batch_size = 2
src_len = 5
tgt_len = 6
vocab_size = 20

# toy input
src = torch.randint(0, vocab_size, (batch_size, src_len))
tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

print("input shape:", src.shape)

model = MyTransformer(token_num=vocab_size)

out = model(src, tgt)

print("Output shape:", out.shape)
